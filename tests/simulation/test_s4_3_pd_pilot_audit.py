import numpy as np

from scripts.simulation.audit_s4_3_pd_pilot import transition_counts


def test_contact_transition_counts_are_directional():
    counts = np.asarray([0, 0, 2, 3, 0, 1, 0])
    assert transition_counts(counts) == (2, 2)


def test_contact_transition_counts_handle_empty_contact():
    assert transition_counts(np.zeros(8)) == (0, 0)
