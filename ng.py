import nevergrad as ng
import subprocess

def pytorchganzoo(x, test=False):
    if test:
      outputs = subprocess.check_output(["/private/home/oteytaud/pytorchganzoo/pytorchganzoo.sh"] + [str(y) for y in x]) #, stdout=subprocess.PIPE)
    else:
      outputs = subprocess.check_output(["/private/home/oteytaud/pytorchganzoo/pytorchganzoo.sh"] + [str(y) for y in x]) #, stdout=subprocess.PIPE)
    rline = outputs.split(b'\n')
#    print(rline)
    for r in rline:
      try:
       res = -float(r)
      except:
       pass
    return res


dim=3
for budget in [10, 20, 40, 80, 160]:
  for r in [5]:
#  for tool in ["OnePlusOne", "RandomSearch", "DiagonalCMA", "TwoPointsDE", "DE", "PSO", "SQP"]:
   for tool in ["RandomSearch", "ScrHammersleySearch", "LHSSearch", "DE", "DiagonalCMA", "CMA"]:
    optimizer = ng.optimizers.registry[tool](instrumentation=dim, budget=budget)
    recommendation = optimizer.optimize(pytorchganzoo)
    #print(recommendation)  # optimal args and kwargs
    traine=pytorchganzoo(recommendation.data)
    teste=pytorchganzoo(recommendation.data, test=True)
    print(budget, r, tool, traine, teste, "#results") #, traine+26.282498, teste+23.751754, "#results")


#./fitness_test.sh 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#~/pytorchganzoo/codes ~/pytorchganzoo
#    odd PSNR_Y: 26.282498 dB; SSIM_Y: 0.661995   even PSNR_Y: 23.751754 dB; SSIM_Y: 0.677056
#    PSNR to be maximized
#    SSIM to be maximized
#    ~/pytorchganzoo
#     23.751754
