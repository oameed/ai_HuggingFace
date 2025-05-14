###############################################
### AI PLAYGROUND                           ###
### INFERENCE AND TRAINIG USING HUGGINGFACE ###
### by: OAMEED NOAKOASTEEN                  ###
############################################### 

import os
import torch as tc

def initialize_run():
  # params[0][0]: model name
  # params[1][0]: num_inference_steps
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('-n', type = str, help = "model name"         , required = True)
  parser.add_argument('-s', type = int, help = "num_inference_steps", default  = 1   )
  args   = parser.parse_args()
  params = [[args.n],
            [args.s] ]
  paths  = [os.path.join(".","data"       ),
            os.path.join(".","predictions") ]
  device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
  return params, paths, device

def get_data(paths):
  def get_filenames(path):
    return [fn for fn in os.listdir(path) if not fn.startswith(".") and not fn.split(".")[1]=="txt"]
  def read_file(filename):
    with open(filename,"r") as fobj:
      string = fobj.read()
    return string
  from diffusers.utils import load_image
  filenames          = get_filenames(paths[0])
  filenames_full     = [os.path.join(paths[0],fn) for fn in filenames]
  filenames_txt      = [x.split(".")[0]+"."+"txt" for x in filenames]
  filenames_txt_full = [os.path.join(paths[0],fn) for fn in filenames_txt]
  images             = [load_image(fn).convert("RGB") for fn in filenames_full]
  prompt             = [read_file(fn) for fn in filenames_txt_full]   
  return images, prompt, filenames

def main():
  from diffusers import FluxFillPipeline
  
  params, paths , device = initialize_run()
  imgs  , prompt, fns    = get_data(paths)
  pipe    = FluxFillPipeline.from_pretrained(params[0][0]                                ,
                                             use_safetensors = True                      ,
                                             cache_dir       = os.environ.get("HF_HOME" ),
                                             token           = os.environ.get("HF_TOKEN") ).to(device)
  
  predict = lambda x,y: pipe(image               = x           ,
                             prompt              = y           ,
                             strength            = 0.4         ,
                             guidance_scale      = 7.5         ,
                             num_inference_steps = params[1][0] )
  
  print("Inference Started ... ")
  predictions = [predict(x,y).images for x,y in zip(imgs,prompt)]
  print("Inference Finished... ")
  
  for i,fn in enumerate(fns):
    predictions[i].save(os.path.join(paths[1],fn))
    
  print("Finished Inference")  

if __name__ == "__main__":
  main()

