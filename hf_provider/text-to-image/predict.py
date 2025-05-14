###############################################
### AI PLAYGROUND                           ###
### INFERENCE AND TRAINIG USING HUGGINGFACE ###
### by: OAMEED NOAKOASTEEN                  ###
############################################### 

import os

def initialize_run():
  # params[0][0]: model name
  # params[1][0]: inference provider
  # params[2][0]: num_inference_steps
  # params[3][0]: output file format
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('-n', type = str, help = "model name"         , required = True )
  parser.add_argument('-p', type = str, help = "inference provider" , required = True )
  parser.add_argument('-s', type = int, help = "num_inference_steps", default  = 1    )
  parser.add_argument('-f', type = str, help = "output file format" , default  = "png")
  args   = parser.parse_args()
  params = [[args.n],
            [args.p],
            [args.s],
            [args.f] ]
  paths  = [os.path.join(".","data"       ),
            os.path.join(".","predictions") ]
  return params, paths

def get_data(paths):
  def get_filenames(path):
    return [fn for fn in os.listdir(path) if not fn.startswith(".")]
  def read_file(filename):
    with open(filename,"r") as fobj:
      string = fobj.read()
    return string
  filenames      = get_filenames(paths[0])
  filenames_full = [os.path.join(paths[0],fn) for fn in filenames]
  prompt         = [read_file(fn) for fn in filenames_full]  
  return prompt, filenames

def main():
  from huggingface_hub import InferenceClient
  
  params, paths = initialize_run()
  prompt, fns   = get_data(paths)
  client  = InferenceClient(model    = params[0][0]              , 
                            provider = params[1][0]              ,
                            token    = os.environ.get("HF_TOKEN") )
  
  predict = lambda x: client.text_to_image(prompt              = x           ,
                                           guidance_scale      = 7.5         ,
                                           num_inference_steps = params[2][0] )
  
  print("Inference Started ... ")
  predictions = [predict(x) for x in prompt] 
  print("Inference Finished... ")
  
  for i,fn in enumerate(fns):
    predictions[i].save(os.path.join(paths[1],fn.split(".")[0]+"."+params[3][0]))
  
  print("Finished!")  
  

if __name__ == "__main__":
  main()

