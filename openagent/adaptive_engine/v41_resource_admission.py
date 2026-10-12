"""Aggregate declared allocation reservations for publisher V4.1 owners."""
from .resource_ledger import ResourceLedger


def reserve_components(components,*,max_total_bytes,owned):
    """Reserve before construction and release after all dependent owners.

    These are explicit engine allocation budgets. Interpreter, operating-system
    file caches and collective-library internal allocations are outside them.
    """
    if type(components) is not dict or not components:
        raise ValueError('explicit complete component admission required')
    if any(type(name) is not str or not name or type(amount) is not int or amount<0
           for name,amount in components.items()):
        raise ValueError('named nonnegative component allocation budgets required')
    required=sum(components.values())
    if max_total_bytes is None:max_total_bytes=required
    if type(max_total_bytes) is not int or max_total_bytes<=0:
        raise ValueError('positive aggregate publisher allocation budget required')
    if required>max_total_bytes:
        raise MemoryError('complete publisher component reservations exceed aggregate admission')
    ledger=ResourceLedger(max_bytes=max_total_bytes,max_resources=len(components)+1)
    owned.callback(ledger.close)
    try:
        for name,amount in components.items():
            if amount:ledger.reserve(name,amount)
    except BaseException:
        ledger.close();raise
    return ledger


__all__=['reserve_components']
