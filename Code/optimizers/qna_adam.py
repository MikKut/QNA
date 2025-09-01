''' Here will be the optimizer itself in form of subclass of torch.optim.Optimizer
 Saves the first nd second momentums, bias correction and introduces additional mechanism:
 scaling down when the observed gradient variance is high.
 '''