import copy
import io
from contextlib import redirect_stdout

import numpy as np
import pycocotools.mask as mask_util
import torch
import torchvision.ops  # For box_iou
import my_utils as utils  # Assuming my_utils contains all_gather
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Assuming these sklearn metrics are needed for the phenotype evaluation part
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error


class CocoEvaluator:
    def __init__(self, coco_gt, iou_types, phenotype_names=None, phenotype_iou_threshold=0.5):  # MODIFIED
        if not isinstance(iou_types, (list, tuple)):
            raise TypeError(f"This constructor expects iou_types of type list or tuple, instead got {type(iou_types)}")
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

        # --- MODIFICATIONS FOR PHENOTYPE EVALUATION ---
        self.phenotype_names = phenotype_names
        self.phenotype_iou_threshold = phenotype_iou_threshold
        self.phenotype_metrics_results = {}  # To store calculated phenotype metrics
        if self.phenotype_names:
            if not isinstance(self.phenotype_names, (list, tuple)):
                raise TypeError("phenotype_names must be a list or tuple of strings.")
            self.all_gt_phenotypes = []
            self.all_pred_phenotypes = []
        # --- END PHENOTYPE MODIFICATIONS ---

    def update(self, predictions, targets_for_pheno: dict = None):  # MODIFIED SIGNATURE
        img_ids_set = set()  # Use a set to automatically handle unique image IDs for COCO eval part

        # Process standard COCO evaluation types
        for original_id in predictions.keys():
            img_ids_set.add(original_id)

        img_ids_list_for_coco = list(img_ids_set)

        for iou_type in self.iou_types:
            # Ensure results are prepared only for images relevant to this update call
            current_predictions = {img_id: predictions[img_id] for img_id in img_ids_list_for_coco if
                                   img_id in predictions}
            if not current_predictions:
                continue

            results = self.prepare(current_predictions, iou_type)  #
            # Suppress pycocotools print output
            with redirect_stdout(io.StringIO()):
                coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()  #
            coco_eval_instance = self.coco_eval[iou_type]

            coco_eval_instance.cocoDt = coco_dt
            coco_eval_instance.params.imgIds = img_ids_list_for_coco  # Use unique image IDs from current batch

            # The 'evaluate' function called here is the one from pycocotools.cocoeval.COCOeval.evaluate, not the engine's one
            # This part seems to be calling a local 'evaluate' function from the provided coco_eval.py.
            # Let's assume it's the local helper 'evaluate(imgs_cocoeval_obj_to_run_evaluate_on)'
            current_img_ids, current_eval_imgs = evaluate(
                coco_eval_instance)  # # evaluate is a helper at the end of this file

            self.eval_imgs[iou_type].append(current_eval_imgs)
            # Accumulate img_ids for synchronization, ensuring uniqueness and order later
            # self.img_ids are extended for COCO eval, not directly used by phenotype matching here

        # Consolidate img_ids from this batch for COCO part, handled in synchronize
        self.img_ids.extend(img_ids_list_for_coco)

        # --- PHENOTYPE MATCHING AND COLLECTION ---
        if self.phenotype_names and targets_for_pheno:
            cpu_device = torch.device("cpu")  # Ensure data is on CPU for numpy conversion
            num_phenotype_features = len(self.phenotype_names)

            for original_id, output_dict in predictions.items():
                if original_id not in targets_for_pheno:
                    continue

                target_dict = targets_for_pheno[original_id]

                try:
                    if not all(k in output_dict for k in ["boxes", "phenotypes"]) or \
                            not all(k in target_dict for k in ["boxes", "phenotypes"]):
                        continue

                    # Ensure data is on CPU
                    gt_boxes = target_dict["boxes"].to(cpu_device)
                    gt_phenotypes = target_dict["phenotypes"].to(cpu_device)

                    pred_boxes = output_dict["boxes"].to(cpu_device)
                    pred_phenotypes = output_dict["phenotypes"].to(cpu_device)

                    num_gt = gt_boxes.shape[0]
                    num_pred = pred_boxes.shape[0]

                    if num_gt == 0 or num_pred == 0:
                        continue

                    # Ensure phenotypes have the correct number of features before proceeding
                    if gt_phenotypes.shape[1] != num_phenotype_features or \
                            pred_phenotypes.shape[1] != num_phenotype_features:
                        # print(f"DEBUG: Phenotype feature count mismatch for image {original_id}. Skipping.")
                        continue

                    iou_matrix = torchvision.ops.box_iou(gt_boxes.float(), pred_boxes.float())
                    matched_pred_indices = set()

                    for gt_idx in range(num_gt):
                        best_iou_for_this_gt = -1.0
                        best_pred_idx_for_this_gt = -1
                        for pred_idx in range(num_pred):
                            if pred_idx in matched_pred_indices:
                                continue
                            current_iou = iou_matrix[gt_idx, pred_idx].item()
                            if current_iou > best_iou_for_this_gt:
                                best_iou_for_this_gt = current_iou
                                best_pred_idx_for_this_gt = pred_idx

                        if best_iou_for_this_gt >= self.phenotype_iou_threshold and best_pred_idx_for_this_gt != -1:
                            gt_pheno_to_add = gt_phenotypes[gt_idx].squeeze().cpu().numpy()
                            pred_pheno_to_add = pred_phenotypes[best_pred_idx_for_this_gt].squeeze().cpu().numpy()

                            # Ensure they are 1D arrays of the correct length
                            if gt_pheno_to_add.ndim == 1 and gt_pheno_to_add.shape[0] == num_phenotype_features and \
                                    pred_pheno_to_add.ndim == 1 and pred_pheno_to_add.shape[
                                0] == num_phenotype_features:
                                self.all_gt_phenotypes.append(gt_pheno_to_add)
                                self.all_pred_phenotypes.append(pred_pheno_to_add)
                                matched_pred_indices.add(best_pred_idx_for_this_gt)
                except Exception as e:
                    # print(f"DEBUG: Error during phenotype matching for image {original_id}: {e}")
                    continue
        # --- END PHENOTYPE MATCHING ---

    def synchronize_between_processes(self):
        # Synchronize standard COCO eval images
        for iou_type in self.iou_types:
            # Ensure eval_imgs for this iou_type is not empty before concatenation
            if self.eval_imgs[iou_type]:
                self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)  #
                create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])  #
            else:  # Handle case where no evaluations were added for an iou_type (e.g. no predictions for this type)
                # Create a dummy eval_imgs or ensure create_common_coco_eval can handle empty
                # For now, if it's empty, we might not need to call create_common_coco_eval
                # Or, ensure img_ids passed to create_common_coco_eval is also filtered accordingly
                # This part needs careful handling based on how create_common_coco_eval behaves with potentially empty/mismatched data
                # print(f"Warning: No eval_imgs to synchronize for iou_type {iou_type}")
                # A simple fix might be to ensure self.img_ids is also filtered or handled if eval_imgs is empty
                # The original create_common_coco_eval merges img_ids globally then filters eval_imgs by unique idx.
                # If an iou_type had no evals, its self.eval_imgs[iou_type] would be empty.
                # It's best if create_common_coco_eval is robust or we ensure it gets valid (even if empty) eval_imgs.
                # For now, let's assume if eval_imgs[iou_type] is empty, we might skip its create_common_coco_eval call,
                # or ensure it's initialized as an empty structure that concatenate and create_common_coco_eval can handle.
                # The original code concatenates then calls create_common_coco_eval. So if list is empty, concatenate fails.
                # So we should only call if list is not empty.
                # The global self.img_ids are gathered in create_common_coco_eval via merge.
                pass

        # --- SYNCHRONIZE PHENOTYPE DATA ---
        if self.phenotype_names:
            if utils.is_dist_avail_and_initialized():  # Check if distributed training
                gathered_gt_lists = utils.all_gather(self.all_gt_phenotypes)
                gathered_pred_lists = utils.all_gather(self.all_pred_phenotypes)

                flat_gt_list = []
                for sublist in gathered_gt_lists:
                    flat_gt_list.extend(sublist)
                self.all_gt_phenotypes = flat_gt_list

                flat_pred_list = []
                for sublist in gathered_pred_lists:
                    flat_pred_list.extend(sublist)
                self.all_pred_phenotypes = flat_pred_list
        # --- END PHENOTYPE SYNCHRONIZATION ---

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()  #

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print(f"IoU metric: {iou_type}")  #
            coco_eval.summarize()  #

        # --- SUMMARIZE PHENOTYPE METRICS ---
        if self.phenotype_names and self.all_gt_phenotypes and self.all_pred_phenotypes:
            self._calculate_and_print_phenotype_metrics()
        elif self.phenotype_names:
            print("Phenotype Regression Metrics (for this fold/evaluation run):")
            print("  Not enough matched ground truth or prediction phenotype data to calculate metrics.")
            for name in self.phenotype_names:  # Ensure dict has keys even if no data
                self.phenotype_metrics_results[name] = {'r2': np.nan, 'rmse': np.nan, 'mape': np.nan, 'nrmse': np.nan}

    def _calculate_and_print_phenotype_metrics(self):
        """Helper function to calculate and print phenotype regression metrics."""
        print("Phenotype Regression Metrics (for this fold/evaluation run):")

        # Initialize all phenotype metrics to NaN
        for name in self.phenotype_names:
            self.phenotype_metrics_results[name] = {'r2': np.nan, 'rmse': np.nan, 'mape': np.nan, 'nrmse': np.nan}

        try:
            all_gt_phenotypes_np = np.array(self.all_gt_phenotypes)
            all_pred_phenotypes_np = np.array(self.all_pred_phenotypes)
        except Exception as e:  # Catch errors during numpy array conversion (e.g. inconsistent shapes)
            print(f"  Error converting phenotype lists to numpy arrays: {e}. Cannot calculate metrics.")
            # Results remain NaN
            return

        basic_structural_validity = False
        if all_gt_phenotypes_np.size > 0 and all_pred_phenotypes_np.size > 0:
            # Check if arrays are 2D and feature counts match, or 1D if only one phenotype feature
            is_gt_shape_ok = (all_gt_phenotypes_np.ndim == 2 and all_gt_phenotypes_np.shape[1] == len(
                self.phenotype_names)) or \
                             (len(self.phenotype_names) == 1 and all_gt_phenotypes_np.ndim == 1 and
                              all_gt_phenotypes_np.shape[0] > 0)
            if len(self.phenotype_names) == 1 and all_gt_phenotypes_np.ndim == 1 and all_gt_phenotypes_np.shape[
                0] == all_gt_phenotypes_np.size:  # Ensure it's not an empty array being misinterpreded
                all_gt_phenotypes_np = all_gt_phenotypes_np.reshape(-1, 1)  # Reshape to 2D for consistent processing
                is_gt_shape_ok = True

            is_pred_shape_ok = (all_pred_phenotypes_np.ndim == 2 and all_pred_phenotypes_np.shape[1] == len(
                self.phenotype_names)) or \
                               (len(self.phenotype_names) == 1 and all_pred_phenotypes_np.ndim == 1 and
                                all_pred_phenotypes_np.shape[0] > 0)
            if len(self.phenotype_names) == 1 and all_pred_phenotypes_np.ndim == 1 and all_pred_phenotypes_np.shape[
                0] == all_pred_phenotypes_np.size:
                all_pred_phenotypes_np = all_pred_phenotypes_np.reshape(-1, 1)  # Reshape to 2D
                is_pred_shape_ok = True

            if is_gt_shape_ok and is_pred_shape_ok and \
                    all_gt_phenotypes_np.shape[0] == all_pred_phenotypes_np.shape[
                0]:  # Number of matched pairs must be same
                basic_structural_validity = True

        gt_is_not_all_zero = False
        if basic_structural_validity:
            if np.any(all_gt_phenotypes_np):
                gt_is_not_all_zero = True

        if basic_structural_validity and gt_is_not_all_zero:
            for i, name in enumerate(self.phenotype_names):
                gt_values = all_gt_phenotypes_np[:, i]
                pred_values = all_pred_phenotypes_np[:, i]

                gt_std = np.std(gt_values)
                if len(gt_values) > 1 and gt_std > 1e-6:
                    try:
                        r2 = r2_score(gt_values, pred_values)
                        rmse = np.sqrt(mean_squared_error(gt_values, pred_values))
                        mape = mean_absolute_percentage_error(gt_values, pred_values)
                        nrmse = rmse / gt_std

                        self.phenotype_metrics_results[name]['r2'] = r2
                        self.phenotype_metrics_results[name]['rmse'] = rmse
                        self.phenotype_metrics_results[name]['mape'] = mape
                        self.phenotype_metrics_results[name]['nrmse'] = nrmse

                        print(
                            f"  {name:<20} R-squared: {r2:.4f}, RMSE: {rmse:.4f}, MAPE: {mape * 100:.2f}%, NRMSE (std): {nrmse:.4f}")
                    except Exception as e:
                        print(f"  {name:<20} Error calculating metrics for this phenotype: {e}")
                else:
                    print(f"  {name:<20} Not enough data or variance for this phenotype to calculate metrics.")
        else:
            print(
                "  Not enough valid matched ground truth or prediction phenotype data to calculate metrics after processing.")
            # phenotype_metrics_results already initialized with NaNs

    def prepare(self, predictions, iou_type):  #
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)  #
        if iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)  #
        if iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)  #
        raise ValueError(f"Unknown iou type {iou_type}")

    def prepare_for_coco_detection(self, predictions):  #
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()  #
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):  #
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks_tensor = prediction["masks"]  # Renamed to avoid conflict with mask_util

            # Ensure masks are boolean before encoding
            masks_bool = masks_tensor > 0.5  #

            scores_list = scores.tolist()  #
            labels_list = labels.tolist()  #

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]  #
                for mask in masks_bool  # Use boolean masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")  #

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels_list[k],
                        "segmentation": rle,
                        "score": scores_list[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):  #
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            # boxes = prediction["boxes"] # Not used in keypoint result structure by default
            # boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()  # category_id
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()  #

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],  # Make sure this is the COCO category ID
                        "keypoints": keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):  #
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_ids, eval_imgs):  #
    # Ensure img_ids is a list before passing to all_gather if it expects a list
    # all_gather typically handles list of tensors or lists of basic python types.
    # Here img_ids is a list of image IDs (ints/strings)
    # eval_imgs is a list of numpy arrays.

    all_img_ids_gath = utils.all_gather(img_ids)  # List of lists of image_ids
    all_eval_imgs_gath = utils.all_gather(eval_imgs)  # List of lists of numpy arrays (or list of numpy arrays)

    merged_img_ids = []
    for p in all_img_ids_gath:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    # all_eval_imgs_gath would be like [ [np_arr1_proc1, np_arr2_proc1,...], [np_arr1_proc2,...] ]
    # but the original code has self.eval_imgs[iou_type].append(eval_imgs), where eval_imgs is already a np.array
    # and then it does np.concatenate(self.eval_imgs[iou_type], 2)
    # So, eval_imgs passed to merge here is already a concatenated numpy array from one process.
    # all_gather(eval_imgs) would then be a list of these numpy arrays, one from each process.
    for p in all_eval_imgs_gath:
        merged_eval_imgs.append(p)  # p is a numpy array here.

    if not merged_img_ids:  # If no image IDs after gathering, perhaps no data from any process
        return np.array([]), np.array([])

    merged_img_ids_np = np.array(merged_img_ids)  # Convert list of IDs to numpy array

    # Concatenate the list of numpy arrays (one from each process)
    # Ensure they can be concatenated (e.g., along existing axis 2 or a new axis if necessary)
    # Original code concatenates along axis 2 *before* merge in synchronize_between_processes.
    # So merged_eval_imgs is a list of arrays that were already concatenated along axis 2 internally per process.
    # We need to concatenate these arrays from different processes.
    # If each element p in merged_eval_imgs has shape e.g. (n_iou_thr, n_area_rng, n_imgs_proc),
    # we'd typically concatenate along the n_imgs_proc dimension (axis 2).
    if not merged_eval_imgs:
        return merged_img_ids_np, np.array([])

    try:
        merged_eval_imgs_np = np.concatenate(merged_eval_imgs, axis=2)  # Concatenate along the images dimension
    except ValueError as e:
        # This might happen if shapes are inconsistent or one process had no evals.
        # print(f"Error during eval_imgs concatenation in merge: {e}. Shapes: {[arr.shape for arr in merged_eval_imgs]}")
        # Fallback or error handling needed. For now, re-raise or return empty if critical.
        # This indicates a deeper issue with how eval_imgs are collected or shaped before merge.
        # For now, let's assume this will work if eval_imgs are correctly formed.
        # If only one process, merged_eval_imgs will have one item, concatenate won't change it if axis=0 and it's just that item.
        # if len(merged_eval_imgs) == 1:
        #    merged_eval_imgs_np = merged_eval_imgs[0]
        # else:
        #    merged_eval_imgs_np = np.concatenate(merged_eval_imgs, axis=2) # Default from source
        # Let's stick to the original paper's concatenate. It should be a list of arrays from all_gather.
        merged_eval_imgs_np = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    unique_merged_img_ids, idx = np.unique(merged_img_ids_np, return_index=True)  #
    # Filter eval_imgs based on these unique indices
    # Ensure merged_eval_imgs_np has content along the axis being indexed by idx
    if merged_eval_imgs_np.size > 0:
        merged_eval_imgs_filtered = merged_eval_imgs_np[..., idx]  #
    else:
        merged_eval_imgs_filtered = np.array([])

    return unique_merged_img_ids, merged_eval_imgs_filtered


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):  #
    # img_ids here is self.img_ids (list of all img_ids from all updates on one process)
    # eval_imgs here is self.eval_imgs[iou_type] (concatenated np array from one process)

    # These are gathered from all processes using utils.all_gather
    img_ids_merged, eval_imgs_merged = merge(img_ids, eval_imgs)  #

    img_ids_list = list(img_ids_merged)
    # eval_imgs_merged is already a numpy array. Flattening it might lose structure.
    # The original code has .flatten() which might be specific to how COCOeval expects evalImgs.
    # COCOeval.evalImgs is a list of dicts. The np.asarray(imgs.evalImgs).reshape(...) in `evaluate` helper
    # suggests evalImgs internally becomes structured.
    # Let's trust the original flatten logic.
    eval_imgs_list = list(eval_imgs_merged.flatten())  #

    coco_eval.evalImgs = eval_imgs_list
    coco_eval.params.imgIds = img_ids_list
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)  #


# This evaluate is a helper specific to this file, not the main engine's evaluate
def evaluate(coco_eval_obj):  # Takes a COCOeval object as input
    # Renamed parameter to avoid confusion with global evaluate
    with redirect_stdout(io.StringIO()):
        coco_eval_obj.evaluate()  # Calls the method of the COCOeval object
    # Reshape needs to be careful if params.imgIds is empty
    num_img_ids = len(coco_eval_obj.params.imgIds)
    if num_img_ids == 0:  # Handle case with no images to evaluate
        # Return empty or appropriately shaped empty arrays
        return coco_eval_obj.params.imgIds, np.array([]).reshape(-1, len(coco_eval_obj.params.areaRng), 0)

    return coco_eval_obj.params.imgIds, np.asarray(coco_eval_obj.evalImgs).reshape(
        -1, len(coco_eval_obj.params.areaRng), num_img_ids
    )  #
