import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F


def latent_reasoning_forward(model, input_ids, attention_mask, reasoning_steps=30):
    """
    Generate text with latent reasoning steps before visible token generation.
    
    Args:
        model: The language model
        input_ids: the tokenized prompt
        reasoning_steps: Number of latent reasoning steps
        max_new_tokens: Maximum number of visible tokens to generate
        
    Returns:
        Generated text with reasoning steps masked
    """
    device = next(model.parameters()).device
    
    # Tokenize the prompt
    B = input_ids.shape[0]
    prompt_length = input_ids.shape[1]
    
    # Get initial embeddings
    embedding_layer = model.get_input_embeddings()
    prompt_embeddings = embedding_layer(input_ids)
    
    # Initialize our tracking variables
    all_embeddings = prompt_embeddings
    all_attention_mask = attention_mask
    
    # Keep track of which positions are latent reasoning steps
    # 0 = prompt token, 1 = latent reasoning, 2 = generated token
    token_types = torch.zeros((1, prompt_length), device=device)  # Start with all prompt tokens
    
    # Phase 1: Latent reasoning steps
    for step in range(reasoning_steps):
        # Forward pass with current embeddings
        outputs = model(
            inputs_embeds=all_embeddings,
            attention_mask=all_attention_mask,
            output_hidden_states=True
        )
        
        # Get the last hidden state
        last_hidden_state = outputs.hidden_states[-1][:, -1:, :]
        
        # Append the hidden state to our embeddings
        all_embeddings = torch.cat([all_embeddings, last_hidden_state], dim=1)
        
        # Update attention mask
        all_attention_mask = torch.cat([
            all_attention_mask,
            torch.ones((B, 1), device=device)
        ], dim=1)
        
        # Mark this position as a latent reasoning step
        token_types = torch.cat([
            token_types,
            torch.ones((1, 1), device=device)  # 1 = latent reasoning
        ], dim=1)
    
    return all_embeddings, all_attention_mask, token_types

def latent_plus_answer_loss(model, embeddings, attention_mask, labels):
    """
    Compute loss for latent reasoning plus answer prediction.
    
    Args:
        model: The language model
        embeddings: Embeddings from latent_reasoning_forward (includes prompt + latent steps)
        attention_mask: Attention mask for embeddings
        labels: Target token IDs to predict
        
    Returns:
        Loss value combining latent reasoning with answer prediction
    """
    device = next(model.parameters()).device
    batch_size = embeddings.shape[0]
    
    # Run the model on the latent embeddings to get predictions
    outputs = model(
        inputs_embeds=embeddings,
        attention_mask=attention_mask,
        output_hidden_states=True
    )
    
    # Get the final hidden states after latent reasoning
    final_hidden_states = outputs.hidden_states[-1]
    
    # Use these hidden states to predict the target labels
    # The logits will be used to compute loss against the target labels
    logits = outputs.logits
    
    # Calculate the loss
    # Shift the logits to match the labels
    # We use the last token prediction to predict the first label token and so on
    shift_logits = logits[:, -1:-1+labels.shape[1], :]
    
    # For cross entropy, we need [B, C, T] for logits and [B, T] for targets
    # where B=batch size, C=vocab size, T=sequence length
    loss = F.cross_entropy(
        shift_logits.transpose(1, 2),  # [B, C, T]
        labels,                        # [B, T]
        ignore_index=-100              # Ignore padding
    )
    
    return loss

def generate_with_latent_reasoning(model, tokenizer, prompt, reasoning_steps=30, max_new_tokens=50):
    """
    Generate text with latent reasoning steps before visible token generation.
    
    Args:
        model: The language model
        tokenizer: The tokenizer for the model
        prompt: Text prompt to generate from
        reasoning_steps: Number of latent reasoning steps
        max_new_tokens: Maximum number of visible tokens to generate
        
    Returns:
        Generated text with reasoning steps masked
    """
    device = next(model.parameters()).device
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    prompt_length = input_ids.shape[1]
    
    # Get initial embeddings
    embedding_layer = model.get_input_embeddings()
    prompt_embeddings = embedding_layer(input_ids)
    
    # Initialize our tracking variables
    all_embeddings = prompt_embeddings
    all_attention_mask = attention_mask
    
    # Keep track of which positions are latent reasoning steps
    # 0 = prompt token, 1 = latent reasoning, 2 = generated token
    token_types = torch.zeros((1, prompt_length), device=device)  # Start with all prompt tokens
    
    # Phase 1: Latent reasoning steps
    for step in range(reasoning_steps):
        # Forward pass with current embeddings
        outputs = model(
            inputs_embeds=all_embeddings,
            attention_mask=all_attention_mask,
            output_hidden_states=True
        )
        
        # Get the last hidden state
        last_hidden_state = outputs.hidden_states[-1][:, -1:, :]
        
        # Append the hidden state to our embeddings
        all_embeddings = torch.cat([all_embeddings, last_hidden_state], dim=1)
        
        # Update attention mask
        all_attention_mask = torch.cat([
            all_attention_mask,
            torch.ones((1, 1), device=device)
        ], dim=1)
        
        # Mark this position as a latent reasoning step
        token_types = torch.cat([
            token_types,
            torch.ones((1, 1), device=device)  # 1 = latent reasoning
        ], dim=1)
    
    # Phase 2: Add </think> token to mark end of reasoning
    think_end_tokens = tokenizer.encode("</think>", add_special_tokens=False)
    if len(think_end_tokens) > 0 and think_end_tokens[0] != tokenizer.unk_token_id:
        think_end_token_id = torch.tensor([[think_end_tokens[0]]], device=device)
        think_end_embedding = embedding_layer(think_end_token_id)
        
        # Add the </think> token embedding
        all_embeddings = torch.cat([all_embeddings, think_end_embedding], dim=1)
        all_attention_mask = torch.cat([
            all_attention_mask,
            torch.ones((1, 1), device=device)
        ], dim=1)
        
        # Mark this as a special token type (using 1.5 to distinguish it)
        token_types = torch.cat([
            token_types,
            torch.ones((1, 1), device=device) * 1.5  # 1.5 = </think> token
        ], dim=1)
    else:
        raise NotImplementedError('think token not in vocabulary')
    
    # Phase 3: Standard autoregressive token generation
    current_length = all_embeddings.shape[1]
    
    # Convert back to input IDs for token generation
    # We'll need to run a forward pass and get the first token
    with torch.no_grad():
        outputs = model(
            inputs_embeds=all_embeddings,
            attention_mask=all_attention_mask
        )
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    
    # Initialize generation with the prompt tokens plus the first generated token
    generated_ids = torch.cat([input_ids, next_token_id], dim=1)
    current_gen_length = generated_ids.shape[1]
    
    # Mark the new token as a generated token
    token_types = torch.cat([
        token_types,
        torch.ones((1, 1), device=device) * 2  # 2 = generated token
    ], dim=1)
    
    # Continue generating tokens
    for i in range(max_new_tokens - 1):  # -1 because we already generated the first token
        with torch.no_grad():
            # We now use standard token generation
            outputs = model(input_ids=generated_ids)
            
            # Get the next token prediction
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Add the token to our sequence
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
            current_gen_length += 1
            
            # Mark this as a generated token
            token_types = torch.cat([
                token_types,
                torch.ones((1, 1), device=device) * 2  # 2 = generated token
            ], dim=1)
            
            # Stop if we hit the EOS token
            if next_token_id.item() == tokenizer.eos_token_id:
                break
    
    # Phase 4: Create the final visible output
    # Extract just the original prompt and the generated tokens (type 0 and 2)
    visible_indices = (token_types == 0) | (token_types == 2)
    
    # We need to convert back to a list of token IDs for decoding
    # First, gather the prompt tokens
    visible_tokens = []
    for i in range(prompt_length):
        visible_tokens.append(input_ids[0, i].item())
    
    # Then add the generated tokens (skipping the latent reasoning indices)
    for i in range(prompt_length, current_gen_length):
        if i - prompt_length >= len(token_types[0]) or token_types[0, i] == 2:
            visible_tokens.append(generated_ids[0, i].item())
    
    # Decode the visible tokens
    visible_text = tokenizer.decode(visible_tokens, skip_special_tokens=True)
    
    # Format the result to indicate reasoning occurred
    # Split the visible text into prompt and response
    # The prompt is the original input provided by the user
    prompt_text = prompt
    
    # Find where the response starts (after the prompt)
    if visible_text.startswith(prompt):
        response_text = visible_text[len(prompt):].strip()
        result = f"{prompt_text} ***thinking*** {response_text}"
    else:
        # Fallback in case there's any issue with exact matching
        result = f"{prompt_text} ***thinking*** {visible_text[len(prompt_text):].strip()}"
    
    return result

# Example usage
def test_latent_reasoning():
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    prompt = "What is the capital of France?"
    result = generate_with_latent_reasoning(
        model, 
        tokenizer, 
        prompt,
        reasoning_steps=5,
        max_new_tokens=30
    )
    
    print(f"Input: {prompt}")
    print(f"Output: {result}")

# Run the test
if __name__ == "__main__":
    test_latent_reasoning()