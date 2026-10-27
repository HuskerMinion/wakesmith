# PyTorch's compiler segfaults when it is first imported after TensorFlow; wakesmith.mww_torch needs it, and other
# tests load TensorFlow first. Importing it here, before any test, keeps the order right for the whole session.
import torch._dynamo  # noqa: F401
