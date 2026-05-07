"""
Training routines and checkpoint management.
"""

import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def train_model_with_checkpoints(model, train_loader, val_loader, criterion, optimizer, scheduler,
                                 num_epochs, initial_epoch=0, patiences=10,
                                 checkpoint_freq=1, checkpoint_dir='./checkpoints',
                                 plot_dir='./training_plots', resume_from=None):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    if resume_from:
        ckpt = torch.load(resume_from, map_location='cpu')
        initial_epoch = ckpt['epoch'] + 1
        train_losses = ckpt['train_losses']
        val_losses = ckpt['val_losses']
        best_val_loss = ckpt['best_val_loss']
        model.load_state_dict(ckpt['model_state_dict'])

    for epoch in range(initial_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        for seq, bio, target in train_loader:
            seq, bio, target = seq.to(model.device), bio.to(model.device), target.to(model.device)
            optimizer.zero_grad()
            out = model(seq, bio)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * seq.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for seq, bio, target in val_loader:
                seq, bio, target = seq.to(model.device), bio.to(model.device), target.to(model.device)
                out = model(seq, bio)
                loss = criterion(out, target)
                val_loss += loss.item() * seq.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        logger.info(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'best_val_loss': best_val_loss,
                'timestamp': datetime.datetime.now().isoformat()
            }, os.path.join(checkpoint_dir, 'best_model.pth'))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patiences:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        if (epoch+1) % checkpoint_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss,
            }, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"))

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model, train_losses, val_losses
