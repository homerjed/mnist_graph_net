from watermark import watermark 
import numpy, scipy, jax, jraph, haiku, optax
print(watermark())
print(watermark(packages="jax,numpy,scipy,haiku,jraph,optax"))
