"""Observe pinned Qwen3.5 returned router fields without route recomputation.

The shared hook driver validates caller-pinned module paths, exact classes and
returned-field indices; it executes only the caller's unchanged publisher model.
"""
from .qwen4_routing_oracle import Qwen4RoutingOracleForward
from .qwen35_publisher_caches import contract


class Qwen35RoutingOracleForward(Qwen4RoutingOracleForward):
    contract_factory=staticmethod(contract)


def make_forward_call(*,options):
    return Qwen35RoutingOracleForward(options)


__all__=['Qwen35RoutingOracleForward','make_forward_call']
