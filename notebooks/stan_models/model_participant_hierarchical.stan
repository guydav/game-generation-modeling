data {
  int<lower=0> N; // how many observations
  int<lower=0> N_val;
  int<lower=1> G; // how many game types
  int<lower=1> P; // the number of participants

  vector[N] attr;
  vector[N] normalized_fitness;
  array[N] int<lower=1, upper=G> game_types;
  array[N] int<lower=1, upper=P> participants;

  vector[N_val] attr_val;
  vector[N_val] normalized_fitness_val;
  array[N_val] int<lower=1, upper=G> game_types_val;
  array[N_val] int<lower=1, upper=P> participants_val;

}

parameters {
  // alpha is subsumed by the participant effect mean
  // real alpha;
  real beta_fitness;
  vector[G - 1] beta;
  real<lower=0> sigma;

  real participant_effect_mean;
  real<lower=0> participant_effect_sigma;
  vector[P] participant_z;
}

transformed parameters {
  vector[P] participant_alpha = participant_effect_mean + participant_effect_sigma * participant_z;

  vector[G] full_beta;
  full_beta[1] = 0;
  for (g in 2:G) {
    full_beta[g] = beta[g - 1];
  }

  vector[N] y = full_beta[game_types] + (beta_fitness * normalized_fitness) + participant_alpha[participants];
  vector[N_val] y_val = full_beta[game_types_val] + (beta_fitness * normalized_fitness_val) + participant_alpha[participants_val];
}

model {
  // alpha ~ normal(0, 1);
  beta_fitness ~ normal(0, 1);
  beta ~ normal(0, 1);
  sigma ~ normal(0, 1);

  participant_effect_mean ~ normal(0, 1);
  participant_effect_sigma ~ normal(0, 1);
  participant_z ~ normal(0, 1);

  attr ~ normal(y, sigma);
}

generated quantities {
  array[N] real attr_pred = normal_rng(y, sigma);
  array[N_val] real attr_val_pred = normal_rng(y_val, sigma);

  vector[N] log_lik;
  vector[N] log_lik_pred;

  for (n in 1:N) {
    log_lik[n] = normal_lpdf(attr[n] | y[n], sigma);
    log_lik_pred[n] = normal_lpdf(attr_pred[n] | y[n], sigma);
  }

  vector[N_val] log_lik_val;
  vector[N_val] log_lik_val_pred;
  for (n in 1:N_val) {
    log_lik_val[n] = normal_lpdf(attr_val[n] | y_val[n], sigma);
    log_lik_val_pred[n] = normal_lpdf(attr_val_pred[n] | y_val[n], sigma);
  }
}
