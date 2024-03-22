import numpy as np
import torch
from tqdm import trange
from Models.pss_predict import MLP
from model import RealNVP, RealNVPBlock
from psssampler import PSSSampler, change_sequence
from train import *
import json
import argparse
import gc

import drjit as dr
import mitsuba as mi
import pyexr


mi.set_variant('cuda_ad_rgb')

N = 8  # Example input dimension
pA = (212, 197)
pB = (208, 185)
pC = (109, 67)

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")


def load_model(filename):
    w_dict = torch.load('logs/' + filename + '.pth')
    model = MLP()
    model.load_state_dict(w_dict['model'])
    return model


pssModel = load_model("best_nee")
for param in pssModel.parameters():
    param.requires_grad = False

pssModel.to(device)

# # v = dr.cuda.ad.TensorXf(5)
# # sequence = dr.cuda.ad.Float(dr.sum(v)) #mi.TensorXf(torch.rand(1, N).to(device))
# mi.register_sampler('pss_sampler', lambda props: PSSSampler(props))

# scene = mi.load_file('./scenes/cbox.xml', res=128, integrator='prb', sampler='pss_sampler')
# params = mi.traverse(scene)
# key = 'sensor.sampler.rng_sequence'
# # params[key] = 0.7
# # img = mi.render(scene, spp=1)
# # pyexr.write('test.exr', img.torch().cpu().numpy())

# 

def do_realnvp(x):
    g_x, _ = realnvp(x)
    return g_x

# @dr.wrap_ad(source='torch', target='drjit')
# def f_A(seq):
#     change_sequence(seq)
#     # params[key] = seq
#     # params.update()
#     img = mi.render(scene, params, spp=1)
#     return img

# @dr.wrap_ad(source='torch', target='drjit')
# def f_B(seq):
#     change_sequence(seq)
#     # params[key] = seq
#     # params.update()
#     # scene = mi.load_file('./scenes/cbox.xml', res=128, integrator='prb', sampler='pss_sampler')
#     img = mi.render(scene, params, spp=1)
#     return img


def f_A(seq):
    x1, y1 = pA
    x1, y1 = (x1 + 0.5) / 512, (y1 + 0.5) / 512
    floats_tensor = torch.tensor([x1, y1]).to(device).repeat(seq.size(0), 1)
    x = torch.cat((floats_tensor, seq), dim=1)
    return pssModel(x)

def f_B(seq):
    x1, y1 = pB
    x1, y1 = (x1 + 0.5) / 512, (y1 + 0.5) / 512
    floats_tensor = torch.tensor([x1, y1]).to(device).repeat(seq.size(0), 1)
    x = torch.cat((floats_tensor, seq), dim=1)
    return pssModel(x)

if __name__ == "__main__":
    #python3 main.py --config config.json  -> To Run the code

    realnvp = RealNVP(N, 8)
    # params = mi.traverse(scene)

    # key = 'red.reflectance.value'

    # # Save the original value
    # param_ref = mi.Color3f(params[key])

    # # Set another color value and update the scene
    # params[key] = mi.Color3f(0.01, 0.2, 0.9)
    # params.update()


    # def mse(image):
    #     return dr.mean(dr.sqr(image[111][67] - img[111][68]))

    
    # opt = mi.ad.Adam(lr=0.05)
    # opt[key] = params[key]
    # params.update(opt)

    # errors = []
    # for it in range(50):
    #     # Perform a (noisy) differentiable rendering of the scene
    #     image = mi.render(scene, params, spp=4)

    #     # Evaluate the objective function from the current rendered image
    #     loss = mse(image)

    #     # Backpropagate through the rendering process
    #     dr.backward(loss)

    #     # Optimizer: take a gradient descent step
    #     opt.step()

    #     # Post-process the optimized parameters to ensure legal color values.
    #     opt[key] = dr.clamp(opt[key], 0.0, 1.0)

    #     # Update the scene state to the new optimized values
    #     params.update(opt)

    #     # Track the difference between the current color and the true value
    #     err_ref = dr.sum(dr.sqr(param_ref - params[key]))
    #     print(f"Iteration {it:02d}: parameter error = {err_ref[0]:6f}", end='\r')
    #     errors.append(err_ref)
    # print('\nOptimization complete.')
    
    #Parse the input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json', help='Specify the config file')
    args = parser.parse_args()

    # Load the configuration from the specified config file
    with open("config.json", "r") as config_file:
        config = json.load(config_file)

    N_EPOCHS = config["epochs"]
    LR = config["learning_rate"]

    optimizer = torch.optim.Adam(realnvp.parameters(), lr=LR)

    realnvp = realnvp.to(device)

    # # Save the original value
    # param_ref = params[key]

    # Set another color value and update the scene
    # params[key] = mi.Color3f(0.8, 0.1, 0.1)
    # params.update()

    # img = mi.render(scene, params, spp=4)
    # pyexr.write('test.exr', img.torch().cpu().numpy())

    x1, y1 = pA
    x2, y2 = pB
    
    realnvp.train()
    for epoch in trange(N_EPOCHS):
        optimizer.zero_grad()
        # Example training data for domain A
         # Sampled from [0, 1)^N

        x = torch.rand(10000, N).to(device)
        g_x = do_realnvp(x)
        # Apply the invertible mapping g to map x_A to domain B
        # Compute f_A(x) and f_B(g(x))
        f_A_x = f_A(x)
        f_B_g_x = f_B(g_x)
        
        # Minimize (f_A(x) - f_B(g(x)))^2
        loss = torch.mean(torch.square(f_A_x - f_B_g_x))
        
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    # realnvp.eval()
    # with torch.no_grad():
    #     x_test = torch.tensor([0.23, 0.25, 0.6, 0.4]).unsqueeze(0).to(device)
    #     g_x, _ = realnvp(x_test)
    # housekeeping
    gc.collect()
    torch.cuda.empty_cache()


