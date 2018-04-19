import pickle
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

plt.set_cmap("jet")

#u3d, v3d： m/s, dx, dy: m
def calc_curl_div(u3d, v3d, dx, dy):
    nx,ny,nz=u3d.shape
    print(u3d.shape)
    print(len([dx,dy]))
    print(len([0,1]))    
    du_dx,du_dy,du_dz = np.gradient(u3d, 1)
    dv_dx,dv_dy,dv_dz = np.gradient(v3d, 1)
    curl3d = dv_dx - du_dy
    div3d = du_dx + dv_dy
    return (curl3d, div3d)
    
#filter scatter points where abs(var3d) < filter_value 
def plot_scatter3d(x1d, y1d, z1d, var3d, filter_value=0, title=""):
    xy_step=1 #to control density of scatter
    z_step=1  #to control dnesity of scatter
    x1d_s=x1d[::xy_step]
    y1d_s=y1d[::xy_step]
    z1d_s=z1d[::z_step]
    var3d_s=var3d[::xy_step, ::xy_step, ::z_step]
    nx=len(x1d_s)
    ny=len(y1d_s)
    nz=len(z1d_s)
    print("(%d,%d,%d)"%(nx,ny,nz))
    x3d,y3d,z3d=np.meshgrid(x1d_s,y1d_s,z1d_s,indexing='ij')
    x3d_to1d=x3d.reshape((nx*ny*nz),order='F')
    y3d_to1d=y3d.reshape((nx*ny*nz),order='F')
    z3d_to1d=z3d.reshape((nx*ny*nz),order='F')
    var3d_to1d=var3d_s.reshape((nx*ny*nz),order='F')
    var3d_to1d_f=var3d_to1d[np.abs(var3d_to1d)>=filter_value]
    x3d_to1d_f=x3d_to1d[np.abs(var3d_to1d)>=filter_value]
    y3d_to1d_f=y3d_to1d[np.abs(var3d_to1d)>=filter_value]
    z3d_to1d_f=z3d_to1d[np.abs(var3d_to1d)>=filter_value]
    absmax_var3d=np.max(np.abs(var3d_to1d_f))
    fig=plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    p=ax.scatter(x3d_to1d_f, y3d_to1d_f, z3d_to1d_f, c=var3d_to1d_f, vmin=-absmax_var3d, vmax=absmax_var3d)
    fig.colorbar(p)
    ax.set_xlabel('Lon')
    ax.set_ylabel('Lat')
    ax.set_zlabel('Z')
    plt.title(title)
    plt.show()


def plot_quiver3d(x1d, y1d, z1d, u3d, v3d, filter_uv_value=0, title=""):
    xy_step=2
    z_step=2
    x1d_s=x1d[::xy_step]
    y1d_s=y1d[::xy_step]
    z1d_s=z1d[::z_step]
    u3d_s=u3d[::xy_step, ::xy_step, ::z_step]
    v3d_s=v3d[::xy_step, ::xy_step, ::z_step]
    nx=len(x1d_s)
    ny=len(y1d_s)
    nz=len(z1d_s)
    print("(%d,%d,%d)"%(nx,ny,nz))
    x3d,y3d,z3d=np.meshgrid(x1d_s,y1d_s,z1d_s,indexing='ij')
    x3d_to1d=x3d.reshape((nx*ny*nz),order='F')
    y3d_to1d=y3d.reshape((nx*ny*nz),order='F')
    z3d_to1d=z3d.reshape((nx*ny*nz),order='F')
    u3d_to1d=u3d_s.reshape((nx*ny*nz),order='F')
    v3d_to1d=v3d_s.reshape((nx*ny*nz),order='F')
    uv3d_to1d=np.sqrt(u3d_to1d**2 + v3d_to1d**2)
    uv3d_to1d_f=uv3d_to1d[uv3d_to1d >= filter_uv_value]
    u3d_to1d_f=u3d_to1d[uv3d_to1d >= filter_uv_value]
    v3d_to1d_f=v3d_to1d[uv3d_to1d >= filter_uv_value]
    x3d_to1d_f=x3d_to1d[uv3d_to1d >= filter_uv_value]
    y3d_to1d_f=y3d_to1d[uv3d_to1d >= filter_uv_value]
    z3d_to1d_f=z3d_to1d[uv3d_to1d >= filter_uv_value]
    fig=plt.figure()
    cmap=plt.get_cmap()
    ax = fig.add_subplot(111, projection='3d')
    cvalue=(z3d_to1d_f-15)/20
    #cvalue=(uv3d_to1d_f)/np.max(uv3d_to1d_f)
    p=ax.quiver(x3d_to1d_f, y3d_to1d_f, z3d_to1d_f, u3d_to1d_f, v3d_to1d_f, 0, color=cmap(cvalue), \
              length=1,  arrow_length_ratio=0.5,  normalize=True)
    plt.title(title)
    ax.set_xlabel('Lon')
    ax.set_ylabel('Lat')
    ax.set_zlabel('Z')
    plt.show()

with open("local_evec5d_xb.pickle","rb") as f:
    dct_xb=pickle.load(f,encoding='latin1')
evec5d_xb=dct_xb["evec5d"]
#print(evec5d_xb)

filter_t=[0.001,0.001,0.0015,0.0015,0.002]  #to add & tune, for every mode
filter_uv=[0.008,0.008,0.01,0.01,0.01,0.01]  #to add & tune, for every mode
filter_curl=[0.002,0.002]  #to add & tune, for every mode
filter_div=[0.002,0.002] #to add & tune, for every mode
imode=0
t3d=evec5d_xb[:,:,:,0,imode]
u3d=evec5d_xb[:,:,:,1,imode]
v3d=evec5d_xb[:,:,:,2,imode]
x1d=np.arange(-18,19,1)
y1d=np.arange(-18,19,1)
z1d=np.arange(14,35,1)
x3d,y3d,z3d=np.meshgrid(x1d,y1d,z1d,indexing='ij')
curl3d, div3d=calc_curl_div(u3d, v3d, 1, 1)


##plot scatter 3d
plot_scatter3d(x1d,y1d,z1d,t3d,filter_t[imode],"T mode%d"%(imode+1))
plot_scatter3d(x1d,y1d,z1d,u3d,filter_uv[imode], "U mode%d"%(imode+1))
plot_scatter3d(x1d,y1d,z1d,v3d,filter_uv[imode], "V mode%d"%(imode+1))
plot_scatter3d(x1d,y1d,z1d,curl3d, filter_curl[imode], "curl mode%d"%(imode+1))
plot_scatter3d(x1d,y1d,z1d,div3d,filter_div[imode], "div mode%d"%(imode+1))

##plot quiver 3d
#plot_quiver3d(x1d, y1d, z1d, u3d, v3d,filter_uv[imode])