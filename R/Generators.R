#AUTHOR: Alexandre Galvão Patriota
#IME-USP

tok <- tok::tokenizer$from_pretrained("gpt2")


############################################################
#Loading the model
############################################################
 Model0 <- GPT(
  block_size = 1024,
  n_embd = 768,
  N_Layers = 12,
  nvoc = 50257,
  Head = 12
)
############################################################
#Updating the model with trained parameters from OPENAI
############################################################

Model0$load_state_dict(state_dict = torch_load("Model-weights.pt") )
Model0  = if (torch::cuda_is_available()) Model0$cuda() else Model0$cpu()


Generate = function(Model=Model0 , max_new_tokens = config$max_new_tokens, temperature=config$temperature, top_k = config$top_k, device0=if (torch::cuda_is_available()) "cuda" else "cpu"){

        raw_text = readline(prompt = "Type your prompt here >> ")
	idx = tok$encode(raw_text)$ids
	cat("\n \n===================== Generating Tokens =====================\n \n")
	cat(raw_text)
	idx = torch::torch_tensor(idx+1, dtype=torch::torch_int(), device=device0)
	idx = torch::torch_unsqueeze(idx, 1)
	torch::with_no_grad({
	for(i in 1:max_new_tokens) {
            if(idx$size(2) <= 1024){ 
                idx_cond = idx
	    } else{
		    k1=idx$size(2)-1024+1; k2 =idx$size(2)
			    idx_cond = idx[,k1:k2]}

            logits = Model$eval()(idx_cond) 
            logits = logits[,min(idx$size(2),1024), ] / temperature
            if(!is.null(top_k)){
                v = torch::torch_topk(logits, min(top_k, logits$size(-1)))
                logits[, -v[[2]][1,]] = -Inf
	    }
            probs = torch::nnf_softmax(logits,-1)
            idx_next = torch::torch_multinomial(probs, num_samples=1)
            idx = torch::torch_cat(list(idx, idx_next), 2)
	    cat(tok$decode(as.integer(idx_next$cpu()-1)))
	    idx_next0 <- as.numeric(idx_next$cpu())
	    if(idx_next0 >= 50256) return(paste("End-of-Sentence \n"))
	    }
	    cat('\n')

	})
}


