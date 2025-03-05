import torch
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


##########################################################################

class Op:
  @staticmethod
  def n_fold_prod(arr : list) -> torch.tensor:
    #assert len(arr) >=2 , "array must contain at least 2 arguments"
    ans = arr[0]
    for i in range(1,len(arr)):
      ans = torch.matmul(ans,arr[i])
    return ans

  @staticmethod
  def ketbra(b_ : list,db : list) -> callable:
    def inner(i : int,j : int) -> torch.tensor:
      return torch.kron(b_[i],db[j])
    return inner

  @staticmethod
  def dag(A : torch.tensor) -> torch.tensor:
    return torch.conj(A.T)

  @staticmethod
  def flip(i,j,phi,n):
    ans = torch.eye(n,dtype = torch.complex64)
    ans[i,i] = np.cos(phi)
    ans[i,j] = - np.sin(phi)
    ans[j,i] = np.sin(phi)
    ans[j,j] = np.cos(phi)
    return ans

  @staticmethod
  def flip_krauss(i,j,phi,n,k):
    ans = torch.eye(n,dtype = torch.complex64)
    ans[i,i] = np.cos(phi)
    ans[i,j] = - np.sin(phi)
    ans[j,i] = np.sin(phi)
    ans[j,j] = np.cos(phi)
    ans[k,k] = 0
    return ans

  @staticmethod
  def symmetrize(A : torch.tensor) -> torch.tensor:
    n,_ = A.shape
    for i in range(n):
      for j in range(i+1,n):
        A[i,j] = torch.conj(A[j,i])
    return A
  
##########################################################################

class Synch_prototype(ABC):
  def __init__(self,n : int, device = "cpu", **kwargs) -> None:
    self.n = n
    self.device = device
    self.db, self.b_ = self.generate_basis_states()
    self.ketbra = Op.ketbra(self.b_,self.db)

  def generate_basis_states(self) -> tuple:
    db = [torch.tensor([1 if i == j else 0  for i in range(self.n)],dtype = torch.complex64).to(self.device) for j in range(self.n)]
    b_ = [torch.tensor([1 if i == j else 0  for i in range(self.n)],dtype = torch.complex64).reshape(-1,1).to(self.device) for j in range(self.n)]
    return db,b_

  def krauss_channel(self,rho : torch.tensor , A1 : torch.tensor,A2 : torch.tensor) -> torch.tensor:
    rho = rho.to(self.device)
    return A1 @ rho @ Op.dag(A1) + A2 @ rho @ Op.dag(A2)

  def unitary_channel(self,rho : torch.tensor,B : torch.tensor ) -> torch.tensor:
    rho = rho.to(self.device)
    return B @ rho @ Op.dag(B)

  @abstractmethod
  def get_krauss(self,**kwargs) -> tuple:
    pass

  @abstractmethod
  def evolve(self,rho : torch.tensor) -> torch.tensor:
    pass
  
##########################################################################

class Synch_prototype_unitary_B(Synch_prototype,ABC):
  def __init__(self,n : int, device = "cpu", **kwargs) -> None:
    super().__init__(n,device,**kwargs)
    self.A1, self.A2, self.B = self.get_krauss(**kwargs)

  def evolve(self,rho : torch.tensor) -> torch.tensor:
    for i in range(self.n):
      if i % 2 == 0:
        rho = self.krauss_channel(rho,self.A1,self.A2)
      else:
        rho = self.unitary_channel(rho,self.B)
    return rho

##########################################################################

class Synch_0(Synch_prototype_unitary_B):
  def get_krauss(self,**kwargs) -> tuple:
    phi1 = kwargs.get("phi1",np.pi/2)
    phi2 = kwargs.get("phi2",np.pi/2)

    A1 = self.ketbra(1,0).to(self.device)
    A2 = Op.n_fold_prod([Op.flip_krauss(i,i+1,phi1,self.n,0) for i in range(1,self.n-1)]).to(self.device)

    B = Op.n_fold_prod([Op.flip(i,i+1,phi2,self.n) for i in range(self.n-2)]).to(self.device)

    return A1, A2, B
  
##########################################################################

def show_results(fid : np.ndarray, pur : np.ndarray,phi1 : np.ndarray, phi2 : np.ndarray) -> plt.Figure:
  m3 = 1
  fig = plt.figure(figsize = (4.5,3.5))
  ph1,ph2 = np.meshgrid(phi1,phi2)
  surf = np.ones(ph1.shape) * 0.975

  for i in range(1,m3+1):
    ax = fig.add_subplot(1,m3,i)

    im = ax.contourf(fid[:,:,i],cmap = "inferno",levels = [0.9] + [0.9 + i * 0.005 for i in range(1,21)], origin = 'lower',extent = [0.4 - 0.005,0.6 + 0.005,0.4- 0.005,0.6 + 0.005],vmin = 0.9,vmax = 1.001,alpha =0.95)
    ax.contour(fid[:,:,i],levels = [0.975],colors = "yellow", origin = 'lower',extent = [0.4 - 0.005,0.6 + 0.005,0.4- 0.005,0.6 + 0.005],vmin = 0.9,vmax = 1)
    ax.text(0.56,0.55, '0.975', fontsize=10, color='yellow')

    fig.colorbar(im,ax = ax, fraction=0.04, pad=0.15)
    ax.set_xlabel(r"$\varphi$")
    ax.set_ylabel(r"$\theta$")
    ax.set_title(r"$\langle$" + f"{i + 1}" +  r"$|\rho_{ABA} |$" + f"{i + 1}"+ r"$\rangle$")

  return fig

def gen_results_for_completely_mixed(n : int, model_selected : object, device : str = "cpu") -> plt.Figure:
  phi1 = np.arange(0.4, 0.6, 0.001)
  phi2 = np.arange(0.4, 0.6, 0.001)
  fid = np.zeros((len(phi1),len(phi2),n))
  pur = np.zeros((len(phi1),len(phi2)))
  m1,m2,m3 = fid.shape
  rho = torch.eye(n,dtype = torch.complex64) / n

  for i in range(m1):
    for j in range(m2):
      model = model_selected(n,device,**{"phi1" : phi1[i] * np.pi, "phi2" : phi2[j] * np.pi})
      rho_t = model.evolve(rho)
      for k in range(n):
        a = (model.db[k] @ rho_t @ model.b_[k]).item()
        fid[i,j,k] = a.real
      pur[i,j] = torch.trace(rho_t @ rho_t).item().real

  return show_results(fid,pur,phi1,phi2)

