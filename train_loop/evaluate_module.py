import torch
from .utils.model_utils import move_to_device
from tqdm import tqdm

def evaluate_loss(model, val_dataloader, loss_function, device, num_val_steps=-1, use_tqdm=True):
    """
    Evaluate the model on the validation dataset.
    
    Args:
        model: The model to evaluate
        val_dataloader: DataLoader for the validation dataset
        loss_function: Loss function for evaluation
        device: Device to run evaluation on
        
    Returns:
        tuple: (average_loss, loss_components_dict)
    """
    total_loss = 0.0
    loss_components = {}
    num_batches = 0

    dataset = val_dataloader.dataset
    i = 0 

    if use_tqdm:
        bar = tqdm(val_dataloader, desc="Validation")
    else:
        bar = val_dataloader

    with torch.no_grad():
        for batch_inputs_dict in bar:
            if batch_inputs_dict is None:
                continue

            if num_val_steps > 0 and i > num_val_steps:
                break
        
            i += 1
                
            batch_inputs_dict = move_to_device(batch_inputs_dict, device)
            
            # Get teacher outputs
            if hasattr(dataset , 'post_process_batch'):
                dataset.post_process_batch(batch_inputs_dict)
            
            # Get outputs
            model_outputs = model(batch_inputs_dict)
            
            # Calculate loss
            loss, losses = loss_function(batch_inputs_dict, model_outputs)
            
            total_loss += loss.item()
            
            # Track individual loss components
            for k, v in losses.items():
                if k not in loss_components:
                    loss_components[k] = 0.0
                loss_components[k] += v.item()
                
            num_batches += 1
    
    # Calculate averages
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_components = {k: v / num_batches for k, v in loss_components.items()} if num_batches > 0 else {}
    
    model.train()
    avg_components['total_loss'] = avg_loss
    return avg_loss, avg_components
