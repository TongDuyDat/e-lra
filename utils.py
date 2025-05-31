import torch
import math

def check_loss_nan(loss, verbose=True):
    """
    Checks if the provided loss (PyTorch tensor or float) contains NaN values.

    Parameters:
    - loss: Union[float, torch.Tensor]
        The loss value or tensor to be checked for NaNs.
    - verbose: bool (default=True)
        If True, prints a warning message when a NaN is detected.

    Returns:
    - bool
        True if NaN is detected, False otherwise.
    """
    if isinstance(loss, torch.Tensor):
        if torch.isnan(loss).any():
            if verbose:
                print("[WARNING] Loss contains NaN values (PyTorch Tensor).")
            return True
        return False
    elif isinstance(loss, float):
        if math.isnan(loss):
            if verbose:
                print("[WARNING] Loss is NaN (float value).")
            return True
        return False
    else:
        raise TypeError(
            f"Unsupported loss type: {type(loss)}. Expected float or torch.Tensor"
        )


def check_target_range(target_tensor, name="target"):
    """
    Kiểm tra tensor target có chứa giá trị ngoài khoảng [0, 1] hay không.

    Args:
        target_tensor (torch.Tensor): Tensor cần kiểm tra.
        name (str): Tên tensor để hiển thị trong thông báo lỗi.

    Raises:
        ValueError: Nếu phát hiện giá trị nằm ngoài khoảng [0, 1].
    """
    if not isinstance(target_tensor, torch.Tensor):
        raise TypeError("Đầu vào phải là một torch.Tensor")

    min_val = target_tensor.min()
    max_val = target_tensor.max()

    if min_val < 0 or max_val > 1:
        return True
    else:
        return False
