import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ..data.dataset import ReplayBuffer
from ..models.teacher import TeacherModel
from ..models.student import StudentModel
import torch.nn.functional as F

def distillation_loss(student_outputs, teacher_wdl, teacher_margin, teacher_policy_probs, teacher_policy_indices, game_result):
    # WDL loss (distill from teacher + hard game result)
    # BUILD_SPEC 5.5: w_wdl_distill * kl_div(student_wdl, teacher_wdl) + w_game * cross_entropy(student_wdl, game_result)
    
    # student_outputs['wdl'] is [B, 3]
    # teacher_wdl is [B, 3]
    # teacher_policy_probs is sparse, this makes policy distillation tricky in a simple batch loop.
    
    loss_wdl_distill = F.kl_div(torch.log(student_outputs['wdl'] + 1e-8), teacher_wdl, reduction='batchmean')
    
    # game_result is 0..4 (white_road, black_road, white_flat, black_flat, draw)
    # map to 3 classes: white_win (0,2), draw (4), black_win (1,3)
    # BUILD_SPEC says teacher_wdl is win, draw, loss from white perspective.
    target_wdl_hard = torch.zeros_like(teacher_wdl)
    for i, res in enumerate(game_result):
        if res in [0, 2]: target_wdl_hard[i, 0] = 1.0 # White win
        elif res == 4: target_wdl_hard[i, 1] = 1.0 # Draw
        elif res in [1, 3]: target_wdl_hard[i, 2] = 1.0 # Black win
        
    loss_wdl_hard = F.cross_entropy(student_outputs['wdl'], target_wdl_hard)
    
    loss_margin_distill = F.mse_loss(student_outputs['margin'], teacher_margin.unsqueeze(1))
    
    # Policy distillation: for now we skip because of sparse indexing complexity in this loop
    # In a full implementation, we would gather the student logits for the specific teacher_policy_indices
    # and compute KL divergence.
    
    return 0.7 * loss_wdl_distill + 0.3 * loss_wdl_hard + 0.3 * loss_margin_distill

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        board_tensor = batch['board_tensor'].to(device)
        size_id = batch['size_id'].to(device)
        teacher_wdl = batch['teacher_wdl'].to(device)
        teacher_margin = batch['teacher_margin'].to(device)
        game_result = batch['game_result'].to(device)
        
        optimizer.zero_grad()
        outputs = model(board_tensor, size_id)
        
        loss = distillation_loss(outputs, teacher_wdl, teacher_margin, batch['policy_probs'], batch['policy_indices'], game_result)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(dataloader)

import time

def benchmark_dataloader(dataloader, num_steps=100):
    start = time.time()
    count = 0
    for i, batch in enumerate(dataloader):
        count += len(batch['board_tensor'])
        if i >= num_steps:
            break
    end = time.time()
    fps = count / (end - start)
    print(f"Dataloader throughput: {fps:.2f} positions/sec")
    return fps

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Placeholder for manifest and config
    # manifest = Manifest("data/manifests/main.txt")
    # buffer = ReplayBuffer(manifest)
    # dataloader = DataLoader(buffer, batch_size=1024, shuffle=True)
    
    # model = StudentModel().to(device)
    # optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    # for epoch in range(10):
    #     loss = train_one_epoch(model, dataloader, optimizer, device)
    #     print(f"Epoch {epoch}, Loss: {loss}")

if __name__ == "__main__":
    main()
