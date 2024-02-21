functions {
  real induced_dirichlet_lpdf(vector c, vector alpha, real phi) {
    int K = num_elements(c) + 1;
    vector[K - 1] sigma = inv_logit(phi - c);
    vector[K] p;
    matrix[K, K] J = rep_matrix(0, K, K);

    // Induced ordinal probabilities
    p[1] = 1 - sigma[1];
    for (k in 2:(K - 1))
      p[k] = sigma[k - 1] - sigma[k];
    p[K] = sigma[K - 1];

    // Baseline column of Jacobian
    for (k in 1:K) J[k, 1] = 1;

    // Diagonal entries of Jacobian
    for (k in 2:K) {
      real rho = sigma[k - 1] * (1 - sigma[k - 1]);
      J[k, k] = - rho;
      J[k - 1, k] = rho;
    }

    return   dirichlet_lpdf(p | alpha)
           + log_determinant(J);
  }
}

data {
  int<lower=0> N;
  int<lower=0> N_val;
  int<lower=1> G;
  int<lower=3> K;  // how many ordinal levels

  vector[N] attr;
  array[N] int<lower=1, upper=K> attr_ordinal;
  vector[N] normalized_fitness;
  array[N] int<lower=1, upper=G> game_types;

  vector[N_val] attr_val;
  array[N_val] int<lower=1, upper=K> attr_ordinal_val;
  vector[N_val] normalized_fitness_val;
  array[N_val] int<lower=1, upper=G> game_types_val;
}

parameters {
  // real alpha;
  real beta_fitness;
  vector[G - 1] beta;
  real<lower=0> sigma;

  ordered[K - 1] cutpoints;
}

transformed parameters {
  vector[G] full_beta;
  full_beta[1] = 0;
  for (g in 2:G) {
    full_beta[g] = beta[g - 1];
  }

  vector[N] y = full_beta[game_types] + (beta_fitness * normalized_fitness);
  vector[N_val] y_val = full_beta[game_types_val] + (beta_fitness * normalized_fitness_val);
}

model {
  // alpha ~ normal(0, 1);
  beta_fitness ~ normal(0, 1);
  beta ~ normal(0, 1);
  sigma ~ normal(0, 1);
  cutpoints ~ induced_dirichlet(rep_vector(1, K), 0);

  attr_ordinal ~ ordered_logistic(y, cutpoints);
}

generated quantities {
  array[N] int<lower=1, upper=K> attr_ordinal_pred;
  vector[N] log_lik;
  vector[N] log_lik_pred;

  for (n in 1:N) {
    attr_ordinal_pred[n] = ordered_logistic_rng(y[n], cutpoints);
    log_lik[n] = ordered_logistic_lpmf(attr_ordinal[n] | y[n], cutpoints);
    log_lik_pred[n] = ordered_logistic_lpmf(attr_ordinal_pred[n] | y[n], cutpoints);
  }

  array[N_val] int<lower=1, upper=K> attr_ordinal_val_pred;
  vector[N_val] log_lik_val;
  vector[N_val] log_lik_val_pred;

  for (n in 1:N_val) {
    attr_ordinal_val_pred[n] = ordered_logistic_rng(y_val[n], cutpoints);
    log_lik_val[n] = ordered_logistic_lpmf(attr_ordinal_val[n] | y_val[n], cutpoints);
    log_lik_val_pred[n] = ordered_logistic_lpmf(attr_ordinal_val_pred[n] | y_val[n], cutpoints);
  }
}
