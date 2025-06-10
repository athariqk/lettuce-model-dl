import torch


class DualTensor:
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        if not (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)):
            raise TypeError("Both x and y must be a tensor.")
        self.x = x # [C, H, W]
        self.y = y # [C, H, W]

    @property
    def shape(self):
        return self.x.shape

    def dim(self):
        return self.x.dim()

    def to(self, *args, **kwargs):
        self.x = self.x.to(*args, **kwargs)
        self.y = self.y.to(*args, **kwargs)
        return self

    def __getitem__(self, index):
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        raise IndexError("DualTensor index out of range (must be 0 or 1).")

    def __len__(self):
        return 2

    def __str__(self):
        return f"DualTensor(x={self.x}, y={self.y})"

    def __repr__(self):
        return f"DualTensor({self.x!r}, {self.y!r})"  # !r uses repr() for the elements

    def __eq__(self, other):
        if isinstance(other, DualTensor):
            return torch.equal(self.x, other.x) and torch.equal(self.y, other.y)
        if isinstance(other, tuple) and len(other) == 2 and \
                isinstance(other[0], torch.Tensor) and isinstance(other[1], torch.Tensor):
            return torch.equal(self.x, other[0]) and torch.equal(self.y, other[1])
        return False

    def __hash__(self):
        raise TypeError(f"Unhashable type: '{self.__class__.__name__}' as it contains mutable tensors.")

    @classmethod
    def collate(self, list_of_dual_tensors: list['DualTensor']) -> 'DualTensor':
        """
        Collates a list of single-sample DualTensor objects into a single batched DualTensor.
        Each DualTensor in the list is assumed to represent a single sample (e.g., tensor_a has shape [C, H, W]).
        Args:
            list_of_dual_tensors (list[DualTensor]): A list of DualTensor instances.
        Returns:
            DualTensor: A new DualTensor instance where tensor_a and tensor_b are batched.
        Raises:
            ValueError: If the input list is empty.
        """
        if not list_of_dual_tensors:
            raise ValueError("Cannot collate an empty list of DualTensors.")

        # Stack the corresponding tensors from each DualTensor in the list
        # torch.stack will add a new dimension at dim=0 (the batch dimension)
        batched_tensor_a = torch.stack([dt.x for dt in list_of_dual_tensors], dim=0)
        batched_tensor_b = torch.stack([dt.y for dt in list_of_dual_tensors], dim=0)

        return self(batched_tensor_a, batched_tensor_b)


class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        data, target = self.subset[index]

        if self.transform:
            data, target = self.transform(data, target)
            if isinstance(data, list) and len(data) == 2:
                return DualTensor(data[0], data[1]), target

        return data, target

    def __len__(self):
        return len(self.subset)

    @property
    def dataset(self):
        return self.subset.dataset

    @property
    def indices(self):
        return self.subset.indices
