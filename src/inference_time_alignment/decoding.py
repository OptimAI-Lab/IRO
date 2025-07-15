import torch

def get_diverse_beams_1(responses, beam_scores, unfinished_sequences, w, k):
    """
    Get diverse beam outputs by considering unique responses first.
    
    Args:
        responses: Generated response sequences
        scorer: Scoring function for responses
        unfinished_sequences: Binary tensor indicating incomplete sequences
        w: Number of beams to select
        k: Number of successors per beam
    
    Returns:
        Selected beam indices
    """
    
    # Map responses to their indices and scores
    response_indices = {}
    for i, resp in enumerate(responses):
        if resp not in response_indices:
            response_indices[resp] = {'indices': [], 'max_score': float('-inf')}
        response_indices[resp]['indices'].append(i)
        response_indices[resp]['max_score'] = max(
            response_indices[resp]['max_score'], 
            beam_scores[i].item()
        )
    
    # Sort unique responses by their max scores
    sorted_unique = sorted(
        response_indices.items(), 
        key=lambda x: x[1]['max_score'], 
        reverse=True
    )
    
    # Select beams
    selected_indices = []
    # First add highest scoring index from each unique response
    for response, data in sorted_unique:
        if len(selected_indices) < w:
            max_score_idx = max(
                data['indices'], 
                key=lambda i: beam_scores[i].item()
            )
            selected_indices.append(max_score_idx)
    
    # If we need more beams, fill with highest scoring duplicates
    if len(selected_indices) < w:
        remaining = w - len(selected_indices)
        # Get all indices sorted by score, excluding already selected
        all_sorted = sorted(
            [(i, beam_scores[i].item()) for i in range(len(responses))
             if i not in selected_indices],
            key=lambda x: x[1],
            reverse=True
        )
        selected_indices.extend([idx for idx, _ in all_sorted[:remaining]])
    
    # Convert to tensor and repeat for successors
    beam_idx = torch.tensor(selected_indices, device=beam_scores.device)
    return beam_idx.repeat(k)

def get_group_beams(responses, beam_scores, unfinished_sequences, w, k):
    """Sort scores within each W-sized group and select best."""
    
    beam_scores = beam_scores.view(w, k)
    _, beam_idx = torch.topk(beam_scores, 1, dim=1)
    beam_idx = beam_idx.view(-1)
    beam_idx = beam_idx + torch.arange(0, w * k, k, device=beam_scores.device)
    
    return beam_idx.repeat_interleave(k)


def get_beams_probability(responses, beam_scores, unfinished_sequences, w, k, temperature=1.0):
    # Get unique responses and their indices
    unique_responses = {}
    for i, response in enumerate(responses):
        if response not in unique_responses:
            unique_responses[response] = i

    unique_indices = list(unique_responses.values())
    
    # Calculate probabilities for unique beams
    unique_scores = beam_scores[unique_indices]
    unique_scores = unique_scores - torch.max(unique_scores)
    unique_scores = unique_scores / temperature
    probs = torch.softmax(unique_scores, dim=0)
    
    # Sample indices based on probabilities
    selected_indices = torch.multinomial(probs, num_samples=w*k, replacement=True)
    
    # Map back to original indices and repeat for k beams
    beam_idx = torch.tensor([unique_indices[i] for i in selected_indices])
    # beam_idx = beam_idx.repeat(k)
    
    return beam_idx