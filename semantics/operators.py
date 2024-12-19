import operator
from typing import Any, Callable

from .types import (
    PyArguments,
    PyGenericAlias,
    PyInstanceBase,
    PyLiteralType,
    PyType,
    PyUnionType,
)

UNARY_OP_MAP: dict[str, str] = {
    "+": "__pos__",
    "-": "__neg__",
    "~": "__invert__",
}

UNARY_OP_FUNCS: dict[str, Callable[[Any], Any]] = {
    "+": operator.pos,
    "-": operator.neg,
    "~": operator.invert,
}

BINARY_OP_MAP: dict[str, str] = {
    "+": "__add__",
    "-": "__sub__",
    "*": "__mul__",
    "/": "__truediv__",
    "//": "__floordiv__",
    "%": "__mod__",
    "@": "__matmul__",
    "**": "__pow__",
    "<<": "__lshift__",
    ">>": "__rshift__",
    "&": "__and__",
    "^": "__xor__",
    "|": "__or__",
}

BINARY_OP_FUNCS: dict[str, Callable[[Any, Any], Any]] = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "//": operator.floordiv,
    "%": operator.mod,
    "@": operator.matmul,
    "**": operator.pow,
    "<<": operator.lshift,
    ">>": operator.rshift,
    "&": operator.and_,
    "^": operator.xor,
    "|": operator.or_,
}


def get_unary_op_type(op: str, operand: PyType) -> PyType:
    """
    Get the return type of a unary operation.

    Args:
        op: The unary operator.
        operand: The operand type.

    Returns:
        result: The return type of the operation.
    """
    if op not in UNARY_OP_MAP:
        raise ValueError(f"invalid unary operator: {op}")

    if isinstance(operand, PyLiteralType):
        try:
            return PyLiteralType(UNARY_OP_FUNCS[op](operand.value))
        except Exception:
            return PyType.ANY

    if isinstance(operand, PyUnionType):
        return PyUnionType.from_items(
            get_unary_op_type(op, item) for item in operand.items
        )

    if isinstance(operand, PyInstanceBase):
        if method := operand.lookup_method(UNARY_OP_MAP[op]):
            return method.get_return_type()

    return PyType.ANY


def get_binary_op_type(op: str, left: PyType, right: PyType) -> PyType:
    """
    Get the return type of a binary operation.

    Args:
        op: The binary operator.
        left: The left operand type.
        right: The right operand type.

    Returns:
        result: The return type of the operation.
    """
    if op not in BINARY_OP_MAP:
        raise ValueError(f"invalid binary operator: {op}")

    if op == "|" and left.is_annotation() and right.is_annotation():
        return PyGenericAlias.from_stub(
            "typing.Union", (left.get_annotated_type(), right.get_annotated_type())
        )

    if isinstance(left, PyLiteralType) and isinstance(right, PyLiteralType):
        try:
            return PyLiteralType(BINARY_OP_FUNCS[op](left.value, right.value))
        except Exception:
            return PyType.ANY

    if isinstance(left, PyUnionType):
        return PyUnionType.from_items(
            get_binary_op_type(op, item, right) for item in left.items
        )

    if isinstance(right, PyUnionType):
        return PyUnionType.from_items(
            get_binary_op_type(op, left, item) for item in right.items
        )

    if isinstance(left, PyInstanceBase):
        if method := left.lookup_method(BINARY_OP_MAP[op]):
            return method.get_return_type(PyArguments(right))

    if isinstance(right, PyInstanceBase):
        if method := right.lookup_method("__r" + BINARY_OP_MAP[op][2:]):
            return method.get_return_type(PyArguments(left))

    return PyType.ANY
