from torch.utils.data import DataLoader
import torch
import time
import numpy as np
import utils.utils as utils
import copy

def train_local_client_prox(clients, data_loader, epochs, client_id, conn, mu, event):
    train_delay, transfer_delay, batch_size, model, optimizer, loss_fn, host, idx, rounds, device_id = clients[client_id]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.train()
    global_model = copy.deepcopy(model)
    train_round = 0
    while event.is_set() == False:
        pass
    
    while True:
        train_round += 1
        for epoch in range(epochs):
            model_updated = False
            running_loss = 0.0
            
            while conn.poll(): 
                state, selected = conn.recv()
                global_model.load_state_dict(state)
                if selected:
                    model.load_state_dict(state)
                    model_updated = True
                    train_round, tau = 0, 0
                
            if model_updated:
                break
            
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                pred = model(images)
                loss = loss_fn(pred,labels)
                prox_term = 0
                
                # Proximal term (||w_t - w_global||^2)
                for param, global_param in zip(model.parameters(), global_model.parameters()):
                    prox_term += ((param - global_param) ** 2).sum()
                prox_term = (mu / 2) * prox_term
                    
                loss = loss + prox_term
                loss.backward()
                optimizer.step()
                scheduler.step(loss)  # 설정된 step_size 간격으로 학습률 감소
                running_loss += loss.item()
                
            time.sleep(train_delay) # 학습 딜레이
        
        if model_updated == False:
            time.sleep(transfer_delay) # 전송 딜레이
            conn.send((model.state_dict(), running_loss/len(data_loader))) # 부모에게 학습된 모델 전송
            print(f"Client {client_id} - iter [{train_round}], Loss: {running_loss/len(data_loader)}")

def train_local_client_nova(clients, data_loader, epochs, client_id, conn, event):
    train_delay, transfer_delay, batch_size, model, optimizer, loss_fn, host, idx, rounds, device_id = clients[client_id]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.train()
    train_round, tau = 0, 0
    
    while event.is_set() == False:
        pass
    
    while True:
        train_round += 1
        for epoch in range(epochs):
            model_updated = False
            running_loss = 0.0
            
            while conn.poll(): 
                state, selected = conn.recv()
                if selected:
                    model.load_state_dict(state)
                    model_updated = True
                    train_round, tau = 0, 0
                
            if model_updated:
                break
                
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                pred = model(images)
                loss = loss_fn(pred,labels)
                loss.backward()
                optimizer.step()
                scheduler.step(loss)  # 설정된 step_size 간격으로 학습률 감소
                running_loss += loss.item()
                tau += 1
                
            time.sleep(train_delay) # 학습 딜레이
            
        if model_updated == False:
            time.sleep(transfer_delay) # 전송 딜레이
            conn.send((model.state_dict(), tau, running_loss/len(data_loader))) # 부모에게 학습된 모델 전송
            
            print(f"Client {client_id} - iter [{train_round}], Loss: {running_loss/len(data_loader)}, Steps: {tau}")
            
            
def train_local_client_cluster(clients, data_loader, epochs, client_id, conn, n_class, mu, event):
    train_delay, transfer_delay, batch_size, model, optimizer, loss_fn, host, idx, rounds, device_id = clients[client_id]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.train()
    train_round, tau = 0, 0
    all_labels = None
    label_counts = []
    global_model = copy.deepcopy(model)
    
    while event.is_set() == False:
        pass
    
    while True:
        train_round += 1
            
        for epoch in range(epochs):
            model_updated = False
            running_loss = 0.0
            while conn.poll(): 
                state, selected = conn.recv()
                global_model.load_state_dict(state)
                if selected:
                    model.load_state_dict(state)
                    model_updated = True
                    train_round, tau = 0, 0
                
            if model_updated:
                break
            
            for images, labels in data_loader:
                
                images, labels = images.to(device), labels.to(device)
                if all_labels is None:
                    flattened_labels = labels.view(-1)
                    label_counts.append(flattened_labels)
                    
                optimizer.zero_grad()
                pred = model(images)
                loss = loss_fn(pred,labels)
                prox_term = 0
                
                # Proximal term (||w_t - w_global||^2)
                for param, global_param in zip(model.parameters(), global_model.parameters()):
                    prox_term += ((param - global_param) ** 2).sum()
                    
                loss = loss + (mu / 2) * prox_term
                loss.backward()
                optimizer.step()
                # scheduler.step(loss)  # 설정된 step_size 간격으로 학습률 감소
                running_loss += loss.item()
                tau += 1
                
            if all_labels is None:
                all_labels = torch.cat(label_counts)
                all_labels = torch.bincount(input=all_labels, minlength=n_class).cpu().numpy()
            time.sleep(train_delay) # 학습 딜레이
        
        if model_updated == False:
            time.sleep(transfer_delay) # 전송 딜레이
            conn.send((model.state_dict(), tau, running_loss/len(data_loader), all_labels)) # 부모에게 학습된 모델 전송
            # print(f"Client {client_id} - iter [{train_round}], Loss: {running_loss/len(data_loader)}, step: {tau}")