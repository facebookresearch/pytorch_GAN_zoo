import nevergrad as ng
import subprocess

def pgz(x, test=False):
    if test:
      outputs = subprocess.check_output(["/private/home/oteytaud/pytorchganzoo/pytorchganzoo.sh"] + [str(y) for y in x]) #, stdout=subprocess.PIPE)
    else:
      outputs = subprocess.check_output(["/private/home/oteytaud/pytorchganzoo/p[ytorchganzoo.sh"] + [str(y) for y in x]) #, stdout=subprocess.PIPE)
    rline = outputs.split(b'\n')
#    print(rline)
    for r in rline:
      try:
       res = -float(r)
      except:
       pass
    return res


dim=50000
for budget in [1, 10, 100, 1000, 10000]:
 for r in [1]:
  for tool in ["OnePlusOne", "RandomSearch", "DiagonalCMA", "TwoPointsDE", "DE", "PSO", "SQP"]:
    optimizer = ng.optimizers.registry[tool](instrumentation=dim, budget=budget)
    recommendation = optimizer.optimize(pgz)
    #print(recommendation)  # optimal args and kwargs
    traine=pgz(recommendation.data)
    teste=pgz(recommendation.data, test=True)
    print(budget, r, tool, traine, teste, "#results")


#./fitness_test.sh 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#~/pgz/codes ~/pgz
#    odd PSNR_Y: 26.282498 dB; SSIM_Y: 0.661995   even PSNR_Y: 23.751754 dB; SSIM_Y: 0.677056
#    PSNR to be maximized
#    SSIM to be maximized
#    ~/pgz
#     23.751754
