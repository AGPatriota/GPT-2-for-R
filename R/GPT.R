# Code's AUTHOR: Alexandre Galvão Patriota
# IME-USP

nn_gelu_new = torch::nn_module(
    forward = function(x){
      0.5 * x * (1.0 + torch::torch_tanh(sqrt(2.0/pi)*(x + 0.044715 * torch::torch_pow(x, 3.0))))
    }
  )


GPT <- torch::nn_module(
  initialize = function(block_size, n_embd, N_Layers, nvoc, Head, p0 = 0) {
    self$N <- N_Layers
    self$block_size <- block_size
    self$wte <- torch::nn_embedding(nvoc, n_embd)
    self$wpe <- torch::nn_embedding(block_size, n_embd)    
    self$MH <- torch::nn_module_list(lapply(1:N_Layers, function(x) torch::nn_multihead_attention(n_embd, Head, dropout = p0) ))
    self$scale1 <- torch::nn_module_list(lapply(1:N_Layers, function(x) torch::nn_layer_norm(n_embd) ))
    self$scale2 <- torch::nn_module_list(lapply(1:N_Layers, function(x) torch::nn_layer_norm(n_embd) ))
    self$ln_f <- torch::nn_layer_norm(n_embd, elementwise_affine = TRUE)
    self$FFN <- torch::nn_module_list(lapply(1:N_Layers, function(x) torch::nn_sequential(
											  torch::nn_linear(n_embd, 4 * n_embd),
											  nn_gelu_new(),
											  torch::nn_linear(4 * n_embd, n_embd),
											  torch::nn_dropout(p0)
											  ) ))
    self$lm_head <- torch::nn_linear(n_embd, nvoc, bias = FALSE)
    self$drop0 <- torch::nn_dropout(p = p0)
  },
  forward = function(x) {
    x1 <- torch::torch_arange(1, self$block_size, dtype = torch::torch_int(), device = x$device )$unsqueeze(1)
    wei <- torch::torch_zeros(self$block_size, self$block_size,dtype = torch::torch_bool(), device = x$device)
    wei[upper.tri(wei)] <- 1
    if (x$size(2) < self$block_size) {
      zeros <- torch::torch_tensor(rep(50257, self$block_size - x$size(2)), dtype = torch::torch_int(), device = x$device)$unsqueeze(1)$expand(c(x$size(1),-1))
      x <- torch::torch_cat(list(x,zeros), 2)
    }
    output <- self$wte(x) + self$wpe(x1)
    output <- self$drop0(output)
    for (j in 1:self$N) {
      Q <- torch::torch_transpose(self$scale1[[j]](output), 1, 2)
      output <- output + torch::torch_transpose(self$MH[[j]](Q, Q, Q, attn_mask = wei)[[1]], 1, 2  )
      output <- output + self$FFN[[j]](self$scale2[[j]](output))
    }
    output <- self$lm_head(self$ln_f(output))
    output
  }
)
