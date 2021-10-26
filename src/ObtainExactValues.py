import qutip

def obtainExactPurities(N, qstate, partitions, p=0):
    # N denotes the number of qubits
    # qstate is a numpy array of shape  (2**N,) [pure state] or (2**N, 2**N) [mixed state]
    # partitions is a list of partitions
    # p is an optional parameter to admix the the identity to the qstate
    #
    # returns a list of purities corresponding to the list of

    if qstate.shape == (2 ** N,):
        qstate_Q = qutip.Qobj(qstate, dims=[[2] * N, [1] * N])
        qstate_Q = (1 - p) * qstate_Q * qstate_Q.dag() + p / 2 ** N
    else:
        qstate_Q = (1 - p) *qutip.Qobj(qstate, dims=[[2] * N, [2] * N]) + p / 2 ** N

    purities= []
    for partition in partitions:
        rhop = qstate_Q.ptrace(partition)
        purities.append((rhop * rhop).tr())

    return purities