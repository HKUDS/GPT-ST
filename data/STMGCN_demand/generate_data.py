import numpy as np
import h5py


f_graph= h5py.File('../NYC_TAXI/all_graph.h5','r')
f_graph.keys()
print([key for key in f_graph.keys()])
bike_graph_dis = f_graph['dis_bb'][:]
print(f_graph['pcc_bb'][:].shape, f_graph['pcc_bb'][:])
print(f_graph['trans_bb'][:].shape, f_graph['trans_bb'][:])


np.savetxt("dis_bb.csv", f_graph['dis_bb'][:], delimiter=",")
np.savetxt("dis_tt.csv", f_graph['dis_tt'][:], delimiter=",")
np.savetxt("pcc_bb.csv", f_graph['pcc_bb'][:], delimiter=",")
np.savetxt("pcc_tt.csv", f_graph['pcc_tt'][:], delimiter=",")
# A = pd.read_csv("NYC_BIKE.csv", header=None).values.astype(np.float32)
# print(A.shape)
# print(A)
