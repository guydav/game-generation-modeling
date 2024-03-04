data {
  int<lower=0> N; // how many observations
  int<lower=0> N_val; // how many observations
  int<lower=1> G; // how many game types
  int<lower=1> P; // the number of participants
  int<lower=1> U; // the number of unique games

  vector[N] attr;
  vector[N] normalized_fitness;
  array[N] int<lower=1, upper=G> game_types;
  array[N] int<lower=1, upper=P> participants;
  array[N] int<lower=1, upper=U> game_indices;
  array[U] int<lower=1, upper=G> game_types_by_index;

  vector[N_val] attr_val;
  vector[N_val] normalized_fitness_val;
  array[N_val] int<lower=1, upper=G> game_types_val;
  array[N_val] int<lower=1, upper=P> participants_val;
  array[N_val] int<lower=1, upper=U> game_indices_val;
}

parameters {
  real alpha;
  real beta_fitness;
  // vector[G] beta;
  real<lower=0> sigma;

  vector[P] participant_alpha;

  vector[G - 1] game_type_mu;
  // vector[G] game_type_mu;
  vector<lower=0>[G] game_type_sigma;
  vector[U] game_z;
  // vector[U] game_mu;
  // real<lower=0> game_sigma;
}

transformed parameters {
  vector[G] full_game_type_mu;
  full_game_type_mu[1] = 0;
  for (g in 2:G) {
    full_game_type_mu[g] = game_type_mu[g - 1];
  }

  vector[U] game_mu = full_game_type_mu[game_types_by_index] + game_type_sigma[game_types_by_index] .* game_z;
  //  vector[U] game_mu = game_type_mu[game_types_by_index] + game_type_sigma[game_types_by_index] .* game_z;
  vector[N] y = alpha + (beta_fitness * normalized_fitness) + participant_alpha[participants] + game_mu[game_indices];
  vector[N_val] y_val = alpha + (beta_fitness * normalized_fitness_val) + participant_alpha[participants_val] + game_mu[game_indices_val];
}

model {
  alpha ~ normal(0, 1);
  beta_fitness ~ normal(0, 1);

  participant_alpha ~ normal(0, 1);

  sigma ~ normal(0, 1);

  game_type_mu ~ normal(0, 1);
  game_type_sigma ~ normal(0, 1);
  game_z ~ normal(0, 1);

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
