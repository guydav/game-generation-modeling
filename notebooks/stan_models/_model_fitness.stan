data {
  int<lower=0> N;
  vector[N] attr;
  vector[N] normalized_fitness;
}
parameters {
  real alpha;
  real beta;
  real<lower=0> sigma;
}
model {
  alpha ~ normal(0, 1);
  beta ~ normal(0, 1);
  sigma ~ normal(0, 1);
  attr ~ normal(alpha + beta * normalized_fitness, sigma);
}
