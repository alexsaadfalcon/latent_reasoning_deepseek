import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

from utils import format_answer


def latent_reasoning_forward(model, input_ids, attention_mask, reasoning_steps=5):
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
        
        # Check VRAM usage, grows quickly
        # if torch.cuda.is_available():
        #     allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
        #     reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # Convert to GB
        #     print(f"VRAM usage - Step {step}: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
            
        #     # Print shapes to debug memory growth
        #     print(f"Shapes - embeddings: {all_embeddings.shape}, attention_mask: {all_attention_mask.shape}")
    
    return all_embeddings, all_attention_mask, token_types

def latent_reasoning_forward_detach(model, input_ids, attention_mask, reasoning_steps=30):
    """
    Generate text with latent reasoning steps before visible token generation.
    Detach the latent reasoning steps so memory doesn't grow quadratically.
    Each step has gradients based only on the previous step's output.
    
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
    torch.set_grad_enabled(False)
    for step in range(reasoning_steps):
        if step == reasoning_steps - 1:
            torch.set_grad_enabled(True)
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
        
        # Check VRAM usage
        # if torch.cuda.is_available():
        #     allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
        #     reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # Convert to GB
        #     print(f"VRAM usage - Step {step}: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
            
        #     # Print shapes to debug memory growth
        #     print(f"Shapes - embeddings: {all_embeddings.shape}, attention_mask: {all_attention_mask.shape}")
    
    return all_embeddings, all_attention_mask, token_types

def latent_reasoning_forward_one_step_gradients(model, input_ids, attention_mask, reasoning_steps=15):
    """
    Generate text with latent reasoning steps where gradients flow back one step at a time.
    Each step only receives gradients from the immediate next step.
    
    Args:
        model: The language model
        input_ids: the tokenized prompt
        attention_mask: attention mask for the input
        reasoning_steps: Number of latent reasoning steps
        
    Returns:
        Generated text with one-step gradient flow
    """
    device = next(model.parameters()).device
    
    # Tokenize the prompt
    B = input_ids.shape[0]
    prompt_length = input_ids.shape[1]
    
    # Get initial embeddings
    embedding_layer = model.get_input_embeddings()
    prompt_embeddings = embedding_layer(input_ids)
    
    # Initialize our tracking variables
    current_embeddings = prompt_embeddings
    current_attention_mask = attention_mask
    
    # Keep track of which positions are latent reasoning steps
    token_types = torch.zeros((1, prompt_length), device=device)  # Start with all prompt tokens
    
    # Output containers
    all_embeddings = current_embeddings.clone()
    all_attention_mask = current_attention_mask.clone()
    
    # Phase 1: Latent reasoning steps with one-step gradient flow
    for step in range(reasoning_steps):
        # Get current context length
        context_length = current_embeddings.shape[1]
        
        # Forward pass with current embeddings
        outputs = model(
            inputs_embeds=current_embeddings,
            attention_mask=current_attention_mask,
            output_hidden_states=True
        )
        
        # Get the last hidden state
        last_hidden_state = outputs.hidden_states[-1][:, -1:, :]
        
        # Detach all previous embeddings except the most recent one
        # This creates a new computational graph with only one step of history
        if step > 0:
            current_embeddings = torch.cat([
                current_embeddings[:, :-1, :].detach(),  # Detach all but last token
                current_embeddings[:, -1:, :],           # Keep gradients for last token
                last_hidden_state                        # Add new latent state
            ], dim=1)
        else:
            # For first step, keep prompt embeddings connected
            current_embeddings = torch.cat([
                current_embeddings,
                last_hidden_state
            ], dim=1)
        
        # Update attention mask
        current_attention_mask = torch.cat([
            current_attention_mask,
            torch.ones((B, 1), device=device)
        ], dim=1)
        
        # Store the result for return
        if step == 0:
            all_embeddings = current_embeddings.clone()
            all_attention_mask = current_attention_mask.clone()
        else:
            # Only append the new token to avoid memory duplication
            all_embeddings = torch.cat([
                all_embeddings,
                last_hidden_state
            ], dim=1)
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

def construct_logit_mask(label_mask):
    # 9 is the shift from format_answer excluding the end of sentence token
    logit_mask = torch.zeros_like(label_mask)
    logit_mask[:, 9:] = label_mask[:, :-9]
    return logit_mask

def latent_plus_answer_loss(model, embeddings, attention_mask, labels, label_mask):
    """
    Compute loss for latent reasoning plus answer prediction.
    
    Args:
        model: The language model
        embeddings: Embeddings from latent_reasoning_forward (includes prompt + latent steps)
        attention_mask: Attention mask for embeddings
        labels: Target token IDs to predict
        label_mask: Mask for labels
        
    Returns:
        Loss value combining latent reasoning with answer prediction
    """
    device = next(model.parameters()).device
    batch_size = embeddings.shape[0]
    
    # Run the model on the latent embeddings to get predictions
    # outputs = model(
    #     inputs_embeds=embeddings,
    #     attention_mask=attention_mask,
    #     output_hidden_states=True
    # )
    answer_embed = model.get_input_embeddings()(labels)
    combined_embeddings = torch.cat([embeddings, answer_embed], dim=1)
    combined_mask = torch.cat([attention_mask, label_mask], dim=1)
    outputs = model(
        inputs_embeds=combined_embeddings,
        attention_mask=combined_mask,
    )
    
    # Use these hidden states to predict the target labels
    # The logits will be used to compute loss against the target labels
    logits = outputs.logits
    
    # Calculate the loss
    # Shift the logits to match the labels
    # We use the last token prediction to predict the first label token and so on
    shift_logits = logits[:, -(labels.shape[1]+1):-1, :]
    logit_mask = construct_logit_mask(label_mask)
    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    # print(labels[0], tokenizer.decode(labels[0]))
    # print(labels[1], tokenizer.decode(labels[1]))
    # print(logit_mask[0])
    # print(logit_mask[1])
    # input()
    
    # Create masked labels without modifying the original tensor
    masked_labels = labels.clone()
    masked_labels[logit_mask == 0] = -100
    
    # For cross entropy, we need [B, C, T] for logits and [B, T] for targets
    # where B=batch size, C=vocab size, T=sequence length
    loss = F.cross_entropy(
        shift_logits.transpose(1, 2),  # [B, C, T]
        masked_labels,                 # [B, T]
        ignore_index=-100              # Ignore padding
    )
    
    return loss

def generate_with_latent_reasoning(model, tokenizer, prompt, reasoning_steps=5, max_new_tokens=50):
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
    raise NotImplementedError()
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

def generate_with_latent_reasoning_v2(model, tokenizer, prompt, reasoning_steps=5, max_new_tokens=50):
    """
    Generate text with latent reasoning steps before visible token generation.
    Uses the same methodology as in training (latent_reasoning_forward + token generation).
    
    Args:
        model: The language model
        tokenizer: The tokenizer for the model
        prompt: Text prompt to generate from
        reasoning_steps: Number of latent reasoning steps
        max_new_tokens: Maximum number of visible tokens to generate
        
    Returns:
        Generated text with reasoning masked
    """
    device = next(model.parameters()).device
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    
    # Run in inference mode
    with torch.no_grad():
        # Phase 1: Get latent embeddings using latent_reasoning_forward
        all_embeddings, all_attention_mask, token_types = latent_reasoning_forward(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            reasoning_steps=reasoning_steps
        )
        
        # Phase 2: Add </think> token to mark end of reasoning
        think_end_tokens = tokenizer.encode(format_answer(''), add_special_tokens=False)[:-1]
        # print(f"</think> tokens: {think_end_tokens}, decoded: {tokenizer.decode(think_end_tokens)}")
        if len(think_end_tokens) == 9 and think_end_tokens[0] != tokenizer.unk_token_id:
            think_end_token_ids = torch.tensor([think_end_tokens], device=device)
            think_end_embeddings = model.get_input_embeddings()(think_end_token_ids)
            
            # Add the </think> token embeddings
            all_embeddings = torch.cat([all_embeddings, think_end_embeddings], dim=1)
            all_attention_mask = torch.cat([
                all_attention_mask,
                torch.ones((1, 3), device=device)
            ], dim=1)
        else:
            raise NotImplementedError('think token not in vocabulary')
        
        # Phase 3: Generate tokens autoregressively
        generated_ids = []
        
        # Get the starting logits from the latent embeddings
        outputs = model(
            inputs_embeds=all_embeddings,
            attention_mask=all_attention_mask
        )
        next_token_logits = outputs.logits[:, -1, :]
        
        # Start generating tokens
        for i in range(max_new_tokens):
            # Get the next token
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated_ids.append(next_token_id.item())
            
            # Stop if we hit the EOS token
            if next_token_id.item() == tokenizer.eos_token_id:
                break
            
            # Convert token ID to embedding
            next_token_embedding = model.get_input_embeddings()(next_token_id)
            
            # Concatenate to existing embeddings
            all_embeddings = torch.cat([all_embeddings, next_token_embedding], dim=1)
            all_attention_mask = torch.cat([
                all_attention_mask,
                torch.ones((1, 1), device=device)
            ], dim=1)
            
            # Get the next logits for the next iteration
            outputs = model(
                inputs_embeds=all_embeddings,
                attention_mask=all_attention_mask
            )
            next_token_logits = outputs.logits[:, -1, :]
    
    # Keep track of all token types for visibility
    token_types = []
    for i in range(len(input_ids[0])):
        token_types.append(0)  # Prompt tokens
    for i in range(reasoning_steps):
        token_types.append(1)  # Latent reasoning tokens
    for i in range(len(think_end_tokens)):
        token_types.append(2)  # </think> tokens
    for i in range(len(generated_ids)):
        token_types.append(3)  # Generated tokens
    
    # Print token type debug info
    token_counts = {
        0: token_types.count(0),  # Prompt tokens
        1: token_types.count(1),  # Latent reasoning tokens
        2: token_types.count(2),  # </think> tokens
        3: token_types.count(3)   # Generated tokens
    }
    # print(f"Token counts: {token_counts}")
    
    # Decode the visible tokens (prompt + </think> + generated tokens)
    visible_tokens = input_ids[0].tolist() + think_end_tokens + generated_ids
    visible_text = tokenizer.decode(visible_tokens, skip_special_tokens=False)
    # print(f"Full decoded output: {visible_text}")
    
    # Format the result without any extra processing
    return visible_text

def generate_with_latent_reasoning_batch(model, tokenizer, input_ids, attention_mask,
                                         reasoning_steps=5, max_new_tokens=50, temp=0.0, output_attentions=False):
    """
    Generate text with latent reasoning steps in batch mode, returning only the token tensors.
    
    Args:
        model: The language model
        tokenizer: The tokenizer for the model
        input_ids: Tensor of input token IDs [batch_size, seq_len]
        attention_mask: Tensor of attention mask [batch_size, seq_len]
        reasoning_steps: Number of latent reasoning steps
        max_new_tokens: Maximum number of visible tokens to generate
        
    Returns:
        Tensor of generated token IDs [batch_size, seq_len]
    """
    device = next(model.parameters()).device
    batch_size = input_ids.shape[0]
    
    # Run in inference mode
    with torch.no_grad():
        # Phase 1: Get latent embeddings using latent_reasoning_forward
        all_embeddings, all_attention_mask, token_types = latent_reasoning_forward_detach(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            reasoning_steps=reasoning_steps
        )
        
        # Phase 2: Add answer format tokens to mark end of reasoning
        answer_tokens = tokenizer.encode(format_answer(''), add_special_tokens=False)[:-1]
        if len(answer_tokens) == 9 and answer_tokens[0] != tokenizer.unk_token_id:
            answer_token_ids = torch.tensor([answer_tokens], device=device).repeat(batch_size, 1)
            answer_embeddings = model.get_input_embeddings()(answer_token_ids)
            
            # Add the answer format token embeddings
            all_embeddings = torch.cat([all_embeddings, answer_embeddings], dim=1)
            all_attention_mask = torch.cat([
                all_attention_mask,
                torch.ones((batch_size, len(answer_tokens)), device=device)
            ], dim=1)
        else:
            raise NotImplementedError('answer token format not in vocabulary')
        
        # Phase 3: Generate tokens autoregressively
        generated_ids = torch.zeros((batch_size, max_new_tokens), dtype=torch.long, device=device)
        
        # Get the starting logits from the latent embeddings
        outputs = model(
            inputs_embeds=all_embeddings,
            attention_mask=all_attention_mask
        )
        next_token_logits = outputs.logits[:, -1, :]
        
        # Track which samples have completed generation
        eos_token_id = tokenizer.eos_token_id
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Start generating tokens
        for i in range(max_new_tokens):
            # Apply temperature to logits
            if temp == 0:
                next_token_id = torch.argmax(next_token_logits, dim=-1)
            else:
                # Scale logits by temperature
                next_token_logits = next_token_logits / temp
                # Sample from the distribution
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
            
            generated_ids[:, i] = next_token_id
            
            # Mark finished samples
            finished = finished | (next_token_id == eos_token_id)
            if torch.all(finished):
                break
            
            # Convert token IDs to embeddings
            next_token_embedding = model.get_input_embeddings()(next_token_id.unsqueeze(1))
            
            # Concatenate to existing embeddings
            all_embeddings = torch.cat([all_embeddings, next_token_embedding], dim=1)
            all_attention_mask = torch.cat([
                all_attention_mask,
                torch.ones((batch_size, 1), device=device)
            ], dim=1)
            
            # Get the next logits for the next iteration
            outputs = model(
                inputs_embeds=all_embeddings,
                attention_mask=all_attention_mask,
                output_attentions=True,
            )
            next_token_logits = outputs.logits[:, -1, :]
        
        if output_attentions:
            return outputs.attentions, all_embeddings
    
    # Concatenate the input, answer format, and generated tokens
    full_tokens = torch.cat([
        input_ids, 
        answer_token_ids,
        generated_ids[:, :i+1]  # Only include the tokens we've actually generated
    ], dim=1)
    
    return full_tokens

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
