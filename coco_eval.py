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

        self.phenotype_names = phenotype_names
        self.phenotype_iou_threshold = phenotype_iou_threshold
        self.phenotype_metrics_results = {}
        self.all_gt_phenotypes = []
        self.all_pred_phenotypes = []

    def update(self, predictions, targets_for_pheno: dict = None):
        img_ids_set = set()

        # Process standard COCO evaluation types
        for original_id in predictions.keys():
            img_ids_set.add(original_id)

        img_ids_list_for_coco = list(img_ids_set)

        for iou_type in self.iou_types:
            current_predictions = {img_id: predictions[img_id] for img_id in img_ids_list_for_coco if
                                   img_id in predictions}
            if not current_predictions:
                continue

            results = self.prepare(current_predictions, iou_type)  #
            with redirect_stdout(io.StringIO()):
                coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()  #
            coco_eval_instance = self.coco_eval[iou_type]

            coco_eval_instance.cocoDt = coco_dt
            coco_eval_instance.params.imgIds = img_ids_list_for_coco

            current_img_ids, current_eval_imgs = evaluate(
                coco_eval_instance)

            self.eval_imgs[iou_type].append(current_eval_imgs)

        # Consolidate img_ids from this batch for COCO part, handled in synchronize
        self.img_ids.extend(img_ids_list_for_coco)

        # print(f"DEBUG_PRINT: Starting phenotype matching section.")
        # print(f"DEBUG_PRINT:   self.phenotype_names: {self.phenotype_names}")
        # print(f"DEBUG_PRINT:   targets_for_pheno is {'available' if targets_for_pheno else 'None'}")

        if self.phenotype_names and targets_for_pheno:
            # print(f"DEBUG_PRINT: Proceeding with phenotype matching.")
            cpu_device = torch.device("cpu")
            config_expected_num_features = len(self.phenotype_names)
            # print(f"DEBUG_PRINT:   config_expected_num_features: {config_expected_num_features}")

            for original_id, output_dict in predictions.items():
                # print(f"DEBUG_PRINT: Processing phenotypes for original_id: {original_id}")

                if original_id not in targets_for_pheno:
                    print(f"DEBUG_PRINT:   original_id {original_id} not in targets_for_pheno. Skipping.")
                    continue
                # print(f"DEBUG_PRINT:   original_id {original_id} found in targets_for_pheno.")

                target_dict = targets_for_pheno[original_id]

                try:
                    if not all(k in output_dict for k in ["boxes", "phenotypes"]):
                        print(
                            f"DEBUG_PRINT:   Missing 'boxes' or 'phenotypes' in output_dict for {original_id}. Skipping.")
                        continue
                    if not all(k in target_dict for k in ["boxes", "phenotypes"]):
                        print(
                            f"DEBUG_PRINT:   Missing 'boxes' or 'phenotypes' in target_dict for {original_id}. Skipping.")
                        continue
                    # print(
                    #     f"DEBUG_PRINT:   'boxes' and 'phenotypes' keys found in both output_dict and target_dict for {original_id}.")

                    gt_boxes = target_dict["boxes"].to(cpu_device)
                    gt_phenotypes = target_dict["phenotypes"].to(cpu_device)

                    pred_boxes = output_dict["boxes"].to(cpu_device)
                    pred_phenotypes = output_dict["phenotypes"].to(cpu_device)

                    num_gt = gt_boxes.shape[0]
                    num_pred = pred_boxes.shape[0]
                    # print(f"DEBUG_PRINT:   num_gt: {num_gt}, num_pred: {num_pred} for {original_id}")

                    if num_gt == 0 or num_pred == 0:
                        print(
                            f"DEBUG_PRINT:   num_gt or num_pred is 0 for {original_id}. Skipping phenotype matching for this image.")
                        continue

                    # print(
                    #     f"DEBUG_PRINT:   gt_phenotypes shape: {gt_phenotypes.shape}, pred_phenotypes shape: {pred_phenotypes.shape}")

                    if gt_phenotypes.ndim != 2 or pred_phenotypes.ndim != 2:
                        print(
                            f"DEBUG_PRINT:   Phenotype tensors are not 2D for image {original_id}. GT_ndim:{gt_phenotypes.ndim}, Pred_ndim:{pred_phenotypes.ndim}. Skipping.")
                        continue

                    data_num_features_gt = gt_phenotypes.shape[1]
                    data_num_features_pred = pred_phenotypes.shape[1]
                    # print(
                    #     f"DEBUG_PRINT:   data_num_features_gt: {data_num_features_gt}, data_num_features_pred: {data_num_features_pred} for {original_id}")

                    if data_num_features_gt != data_num_features_pred:
                        print(
                            f"DEBUG_PRINT:   Mismatch in feature count between GT ({data_num_features_gt}) and Pred ({data_num_features_pred}) phenotypes for image {original_id}. Skipping.")
                        continue

                    current_image_data_features = data_num_features_gt
                    # print(
                    #     f"DEBUG_PRINT:   current_image_data_features: {current_image_data_features} for {original_id}")

                    if current_image_data_features == 0:
                        print(
                            f"DEBUG_PRINT:   Data has 0 phenotype features for image {original_id} despite instances. Skipping.")
                        continue

                    if current_image_data_features not in [1, 2]:
                        print(
                            f"DEBUG_PRINT:   Data phenotype feature count ({current_image_data_features}) for image {original_id} is not 1 or 2. Skipping.")
                        continue

                    if current_image_data_features != config_expected_num_features:
                        print(
                            f"DEBUG_PRINT:   Data's feature count ({current_image_data_features}) does not match configured "
                            f"feature count ({config_expected_num_features}) for image {original_id}. Skipping.")
                        continue

                    # print(
                    #     f"DEBUG_PRINT:   All phenotype feature checks passed for {original_id}. Proceeding with matching.")

                    iou_matrix = torchvision.ops.box_iou(gt_boxes.float(), pred_boxes.float())
                    matched_pred_indices = set()

                    for gt_idx in range(num_gt):
                        # print(f"DEBUG_PRINT:     Matching for gt_idx: {gt_idx}")
                        best_iou_for_this_gt = -1.0
                        best_pred_idx_for_this_gt = -1
                        for pred_idx in range(num_pred):
                            if pred_idx in matched_pred_indices:
                                continue
                            current_iou = iou_matrix[gt_idx, pred_idx].item()
                            if current_iou > best_iou_for_this_gt:
                                best_iou_for_this_gt = current_iou
                                best_pred_idx_for_this_gt = pred_idx

                        # print(
                        #     f"DEBUG_PRINT:       gt_idx {gt_idx}: best_iou_for_this_gt: {best_iou_for_this_gt}, best_pred_idx_for_this_gt: {best_pred_idx_for_this_gt}")
                        # print(f"DEBUG_PRINT:       self.phenotype_iou_threshold: {self.phenotype_iou_threshold}")

                        if best_iou_for_this_gt >= self.phenotype_iou_threshold and best_pred_idx_for_this_gt != -1:
                            # print(
                            #     f"DEBUG_PRINT:         Match found for gt_idx {gt_idx} with pred_idx {best_pred_idx_for_this_gt} (IoU: {best_iou_for_this_gt:.4f})")
                            gt_pheno_val = gt_phenotypes[gt_idx]
                            pred_pheno_val = pred_phenotypes[best_pred_idx_for_this_gt]

                            gt_pheno_to_add = np.atleast_1d(gt_pheno_val.squeeze().cpu().numpy())
                            pred_pheno_to_add = np.atleast_1d(pred_pheno_val.squeeze().cpu().numpy())
                            # print(
                            #     f"DEBUG_PRINT:           gt_pheno_to_add shape: {gt_pheno_to_add.shape}, pred_pheno_to_add shape: {pred_pheno_to_add.shape}")

                            if gt_pheno_to_add.ndim == 1 and gt_pheno_to_add.shape[0] == current_image_data_features and \
                                    pred_pheno_to_add.ndim == 1 and pred_pheno_to_add.shape[
                                0] == current_image_data_features:
                                # print(f"DEBUG_PRINT:             Phenotype shapes valid. Appending to lists.")
                                self.all_gt_phenotypes.append(gt_pheno_to_add)
                                self.all_pred_phenotypes.append(pred_pheno_to_add)
                                matched_pred_indices.add(best_pred_idx_for_this_gt)
                            else:
                                print(
                                    f"DEBUG_PRINT:             Squeezed phenotype arrays for image {original_id} do not match expected feature count {current_image_data_features}. "
                                    f"GT shape: {gt_pheno_to_add.shape} (ndim {gt_pheno_to_add.ndim}), Pred shape: {pred_pheno_to_add.shape} (ndim {pred_pheno_to_add.ndim}). Skipping this pair.")
                        else:
                            print(f"DEBUG_PRINT:         No sufficient IoU match for gt_idx {gt_idx}.")

                except Exception as e:
                    # print(f"DEBUG_PRINT:   ERROR during phenotype matching for image {original_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        #     print(
        #         f"DEBUG_PRINT:   Finished processing batch. self.all_gt_phenotypes size: {len(self.all_gt_phenotypes)}, self.all_pred_phenotypes size: {len(self.all_pred_phenotypes)}")
        # else:
        #     print(
        #         f"DEBUG_PRINT: Skipping phenotype matching block because self.phenotype_names is falsy or targets_for_pheno is not provided.")


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
                            f"  {name:<20} R-squared: {r2:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}, NRMSE (std): {nrmse:.4f}")
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


def merge(img_ids, eval_imgs):
    all_img_ids_gath = utils.all_gather(img_ids)
    all_eval_imgs_gath = utils.all_gather(eval_imgs)
    merged_img_ids = []
    for p in all_img_ids_gath:
        merged_img_ids.extend(p)
    merged_eval_imgs = []
    for p in all_eval_imgs_gath:
        merged_eval_imgs.append(p)
    if not merged_img_ids:
        return np.array([]), np.array([])
    merged_img_ids_np = np.array(merged_img_ids)
    if not merged_eval_imgs:
        return merged_img_ids_np, np.array([])
    try:
        merged_eval_imgs_np = np.concatenate(merged_eval_imgs, axis=2)
    except ValueError: # Ensure robust concatenation even if one list is empty or shapes differ
        # Fallback: if concatenation fails, attempt to filter and process what's available
        # This might indicate an issue, but we try to proceed if possible.
        # A more robust solution might involve padding or careful pre-checking of shapes.
        # For now, if shapes are truly incompatible for concatenation, this will still fail.
        # A simple case: if only one process contributes, no concatenation is needed.
        if len(merged_eval_imgs) == 1:
            merged_eval_imgs_np = merged_eval_imgs[0]
        else: # Attempt to concatenate valid arrays if some are empty
            valid_arrays = [arr for arr in merged_eval_imgs if arr.size > 0]
            if not valid_arrays: merged_eval_imgs_np = np.array([])
            elif len(valid_arrays) == 1: merged_eval_imgs_np = valid_arrays[0]
            else: merged_eval_imgs_np = np.concatenate(valid_arrays, axis=2)


    unique_merged_img_ids, idx = np.unique(merged_img_ids_np, return_index=True)
    if merged_eval_imgs_np.size > 0:
        # Ensure idx is not out of bounds for the possibly altered merged_eval_imgs_np
        # This requires merged_eval_imgs_np to have its 3rd dimension correspond to original merged_img_ids order
        # If concatenation strategy changed, this indexing might need review. Assuming original intent holds.
        if merged_eval_imgs_np.shape[2] == len(merged_img_ids_np): # Check if 3rd dim matches original merged ids count
             merged_eval_imgs_filtered = merged_eval_imgs_np[..., idx]
        else: # If dimensions don't match up after a fallback concatenation, this filtering is unsafe.
            # print("Warning: Dimension mismatch in merge after eval_imgs concatenation. Results may be compromised.")
            merged_eval_imgs_filtered = np.array([]) # Or handle error more gracefully
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
