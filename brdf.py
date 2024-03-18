# Copyright 2024 Kaining Huang and Tianyi Zhang. All rights reserved.

from typing import TypeAlias, Literal, Optional
import torch
from torch import nn, Tensor
import torch.nn.functional as F

class BRDFModel(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, view_dir: Tensor, normal: Tensor, light_dir: Tensor)->Tensor:
        view_dir = F.normalize(view_dir, dim=0)
        normal = F.normalize(normal, dim=0)
        light_dir = F.normalize(light_dir, dim=0)


# EXPERIMENTAL
class Microfacet(BRDFModel):
    '''
    Specular term
    Reference: GGX Microfacet from Crash Course in BRDF Implementation by Jakub Boksansky
    '''
    def __init__(self, alpha_d: Optional[float] = 0.5, alphd_g: Optional[float] =0.5, f0 = Tensor([0.04, 0.04, 0.04])):
        super().__init__()
        self._alpha_d = nn.Parameter(torch.tensor(alpha_d), requires_grad=True)
        self._alpha_g = nn.Parameter(torch.tensor(alphd_g), requires_grad=True)
        self._f0 = nn.Parameter(f0, requires_grad=False)

    def alpha_d(self): return self._alpha_d
    
    def alpha_g(self): return self._alpha_g

    def ggx_distribution(self, normal: Tensor, half_vector: Tensor)->Tensor:
        '''
        https://boksajak.github.io/files/CrashCourseBRDF.pdf
        Args:
            normal: 
            half_vector:
        '''
        n_dot_h = torch.matmul(normal, half_vector)
        alpha_squared = self.alpha_d() * self.alpha_d()
        denom = (n_dot_h * n_dot_h) * (alpha_squared - 1) + 1
        return alpha_squared / (torch.pi * denom * denom)

    def fresnel_schlick(self, cos_theta: Tensor)-> Tensor:
        # return F.softplus(self._f0) + (1 - F.softplus(self._f0)) * ((1 - cos_theta) ** 5)
        return self._f0 + (1 - self._f0) * ((1 - cos_theta)**5)

    def smith_geometry(self, view_dir, normal, light_dir)->Tensor:
        G1_v = 1/(1+self.lambda_func(view_dir, normal))
        G1_l = 1/(1+self.lambda_func(light_dir, normal))
        return 1 / (1 + G1_v + G1_l)
        # return self.lambda_function(view_dir, normal)*self.lambda_function(light_dir, normal)
    
    def lambda_func(self, v: Tensor, normal: Tensor)->Tensor:
        '''
        https://pbr-book.org/4ed/Reflection_Models/Roughness_Using_Microfacet_Theory
        '''
        dot = torch.sum(v*normal[:, None], dim=0)
        cross = torch.linalg.cross(v, normal[:, None], dim = 0)
        cross_magnitude = torch.linalg.norm(cross, dim = 0)
        tan_theta = cross_magnitude/dot
        auxi = (torch.sqrt(1+self.alpha_g()**2*tan_theta**2)-1)/2
        return auxi

    def microfacet_brdf(self, view_dir: Tensor, normal: Tensor, light_dir: Tensor)->Tensor:
        half_vector = F.normalize(view_dir+light_dir, dim=0)
        d = self.ggx_distribution(normal, half_vector)
        u = torch.relu(torch.sum(view_dir*normal[:, None], dim = 0))
        f = self.fresnel_schlick(u)
        g = self.smith_geometry(view_dir, normal, light_dir)
        return (d * f * g) / (4 * torch.relu(torch.sum(normal[:, None]*view_dir, dim=0)) * torch.relu(torch.sum(normal[:, None]*light_dir, dim=0)))

    def forward(self, view_dir: Tensor, normal: Tensor, light_dir: Tensor)->Tensor:
        view_dir = F.normalize(view_dir, dim=0)
        normal = F.normalize(normal, dim=0)
        light_dir = F.normalize(light_dir, dim=0)
        return self.microfacet_brdf(view_dir, normal, light_dir)


class Lambertian(BRDFModel):
    def forward(self, view_dir: Tensor, normal: Tensor, light_dir: Tensor)->Tensor:
        # Lambertian Cos Law
        normal = F.normalize(normal, dim=-1)
        light_dir = F.normalize(light_dir, dim=-1)
        cos = torch.relu(torch.sum(light_dir*normal, dim = -1))
        return cos


# EXPERIMENTAL
class MicrofacetNeRVDiffuse(BRDFModel):
    '''
    Diffusive only, no parameters to optimize   
    Reference: NeRV: Neural Reflectance and Visibility Fields for Relighting and View Synthesis
    https://arxiv.org/pdf/2012.03927.pdf
    '''
    def __init__(self, f0 = Tensor([0.04, 0.04, 0.04])):
        super().__init__()
        self._f0 = f0

    def get_f0(self): return self._f0

    def forward(self, view_dir: Tensor, normal: Tensor, light_dir: Tensor)->Tensor:
        super().forward(view_dir, normal, light_dir)
        return self.microfacet_nerv_diffusive(view_dir, normal, light_dir)

    def microfacet_nerv_diffusive(self, view_dir: Tensor, normal: Tensor, light_dir: Tensor)->Tensor:
        lambertian_cos = torch.matmul(normal, light_dir)
        half_vector = F.normalize(view_dir+light_dir, dim=0)
        half_vector_cos = torch.matmul(normal, half_vector)
        return lambertian_cos*(1-self.fresnel_schlick(half_vector_cos))
    
    def fresnel_schlick(self, cos_theta: Tensor)-> Tensor:
        return self.get_f0() + (1 - self.get_f0()) * ((1 - cos_theta)**5)
    
# EXPERIMENTAL
class DisneyDiffuse(BRDFModel):
    '''
    https://google.github.io/filament/Filament.html#materialsystem/diffusebrdf
    '''
    def __init__(self, roughness: Optional[float] = 0.0, f0 = torch.tensor(1.0)):
        super().__init__()
        self._roughness = nn.Parameter(torch.tensor(roughness), requires_grad=True)
        self._f0 = f0

    def get_f0(self): return self._f0

    def get_roughness(self): return F.sigmoid(self._roughness)

    def forward(self, view_dir: Tensor, normal: Tensor, light_dir: Tensor)->Tensor:
        view_dir = F.normalize(view_dir, dim=0)
        normal = F.normalize(normal, dim=0)
        light_dir = F.normalize(light_dir, dim=0)

        half_vector = F.normalize(view_dir+light_dir, dim=0)
        NoV = torch.matmul(normal, view_dir)
        NoL = torch.matmul(normal, light_dir)
        LoH = torch.sum(half_vector*light_dir, dim=0)
        return self.Fd_Burley(NoV, NoL, LoH)#*NoL
    
    def F_Schlick(self, u, f90):
        return self.get_f0() + (f90 - self.get_f0()) * torch.pow(1.0 - u, 5.0)

    def Fd_Burley(self, NoV, NoL, LoH):
        f90 = 0.5 + 2.0 * self.get_roughness() * LoH * LoH
        lightScatter = self.F_Schlick(NoL, f90)
        viewScatter = self.F_Schlick(NoV, f90)
        return lightScatter * viewScatter


# EXPERIMENTAL
class Beckmman(BRDFModel):
    def forward(self):
        raise NotImplementedError("Not Implemented")
    
class BRDFFactory:
    Mode: TypeAlias = Literal['Microfacet', 'Lambertian', 'Beckmman', 'Phong']
    @staticmethod
    def get_brdf(BRDF_type: Mode) -> BRDFModel:
        if BRDF_type == "Microfacet-Mono":
            return Microfacet(f0=torch.tensor(0.4))
        elif BRDF_type == "Microfacet-RGB":
            return Microfacet()
        elif BRDF_type == "Lambertian":
            return Lambertian()
        elif BRDF_type == "DisneyDiffuse":
            return DisneyDiffuse(f0=torch.tensor(1.0))
        elif BRDF_type == "NeRVDiffuse":
            return MicrofacetNeRVDiffuse()
        else:
            raise ValueError(f"BRDF type {BRDF_type} not recognized!")


# unit test code
if __name__ == "__main__":
    brdf = BRDFFactory.get_brdf("DisneyDiffuse")

    x = torch.linspace(-1, 1, 50)

    for aa in torch.linspace(-1.0, 1.0, 19):
        s = []
        brdf._roughness.data = aa
        print(aa)
        for xx in x:
            theta = xx*torch.pi/2
            print("SIN: ", torch.sin(theta))
            light_dir = torch.tensor([[torch.abs(torch.sin(theta))],[0],[-torch.cos(theta)]], dtype=float)
            normal = torch.tensor([0,0,-1.0], dtype=float)
            view_dir = torch.tensor([[0],[0],[-.68]], dtype=float)
            ss = brdf(view_dir, normal, light_dir)
            s.append(ss[0])
        
        s = torch.tensor(s)
        import matplotlib.pyplot as plt

        plt.plot(x, s)


    plt.show()