import os
import torch
import torch.nn.functional as F



def train(
        optimizer, 
        scheduler, 
        model, 
        training_dataloader, 
        validation_dataloader,
        num_epochs,
        early_stopping_patience,
        device,
        model_save_root,
        loss_fn=F.mse_loss
    ):

    model.train()
    model_name = model.name
    save_model_name = f"Best_{model_name}.pth"
    os.makedirs(model_save_root, exist_ok=True)
    
    min_valid_loss = float('inf')
    # Training Loop
    avg_train_loss = 10000.
    avg_valid_loss = 10000.
    no_improve = 0
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        
        # Adjust the learning rate
        lr = scheduler(epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        
        step_num = len(training_dataloader)
        
        # Training all batches
        for images, _ in training_dataloader:
            images = images.to(device)
            '''
            TODO: Implement the training loop.
            
            Steps:
            1. Perform the forward pass to obtain the model's output.
            2. Compute the loss based on the difference between the output and the input.
            3. Backpropagate the loss to update the model's parameters.
            '''
            
            optimizer.zero_grad()
            output = model(images)
            loss = loss_fn(output, images)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
        avg_train_loss = sum(train_losses) / len(train_losses)


        # Validation every 3 epochs
        if epoch % 3 == 0:
            model.eval()
            valid_losses = []
            with torch.no_grad():
                for images, _ in validation_dataloader:
                    images = images.to(device)
                    '''
                    TODO: Compute the validation loss.
                    
                    Steps:
                    1. Perform the forward pass on validation data.
                    2. Compute the validation loss.
                    3. Record the loss for analysis.
                    '''
                    output = model(images)
                    loss = loss_fn(output, images)
                    
                    valid_losses.append(loss.item())
                avg_valid_loss = sum(valid_losses) / len(valid_losses)
            
            
            '''
            TODO: Save the model if validation performance improves.
            
            Steps:
            1. Monitor the validation loss for improvement.
            2. Save the model to the specified path if a new best validation loss is achieved.
            '''
            if avg_valid_loss < min_valid_loss:
                min_valid_loss = avg_valid_loss
                no_improve = 0
                save_path = os.path.join(model_save_root, save_model_name)
                torch.save(model.state_dict(), save_path)
            else:
                no_improve += 1
            '''
            TODO: Implement early stopping mechanism.
            
            Steps:
            1. Keep track of validation loss improvements.
            2. Stop training if no improvement is observed for a certain number of checks.
            '''
            if no_improve >= early_stopping_patience:
                print(f"No improvement for {early_stopping_patience} checks. Stopping training.")
                print(f"Epoch: {epoch}, Train Loss: {avg_train_loss}, Valid Loss: {avg_valid_loss}")
                break
            
        print(f"Epoch: {epoch}, Train Loss: {avg_train_loss}, Valid Loss: {avg_valid_loss}")
