data {
  int<lower=0> N;
  int<lower=1> G;
  vector[N] attr;
  array[N] int<lower=1, upper=G> game_types;

}
parameters {
  real alpha;
  vector[G] beta;
  real<lower=0> sigma;
}
model {
  alpha ~ normal(0, 1);

  for (g in 1:G) {
    beta[g] ~ normal(0, 1);
  }

  sigma ~ normal(0, 1);

  for (n in 1:N) {
    attr[n] ~ normal(alpha + beta[game_types[n]], sigma);
  }
}
