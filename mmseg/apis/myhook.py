from mmcv.runner import Hook
import json

class AdapterSelectorControlHook(Hook):

    def __init__(self, warmup_iters=6000):
        self.warmup_iters = warmup_iters
        self.ft_iters = 2000

    def before_train_iter(self, runner):
        current_iter = runner.iter
        if current_iter == 0:
            runner.logger.info(f'[Iter {current_iter}] → Warmup stage, train adapters only')
            runner.model.module.backbone.use_selector = False      
            
            #set all gradients to false of selector
            # runner.model.module.backbone.
            for name, param in runner.model.module.backbone.named_parameters():
                if any(key in name for key in ['selector']): 
                    param.requires_grad = False
                    runner.logger.info(f'[Iter {current_iter}] Freeze {name}')
            
        elif current_iter >= self.warmup_iters:
            #runner.logger.info(f'[Iter {current_iter}] → Start to training selector')
            if current_iter%10 < 7:
                runner.model.module.backbone.use_selector = True
                
                #enable token and selector training
                for name, param in runner.model.module.backbone.named_parameters():
                    if any(key in name for key in ['selector']): 
                        param.requires_grad = True
                        #runner.logger.info(f'[Iter {current_iter}] Train {name}')
        
                #disable adapter
                for name, param in runner.model.module.backbone.named_parameters():
                    if any(key in name for key in ['vlm_adapter', 'vfm_adapter']): 
                        param.requires_grad = False
                        #runner.logger.info(f'[Iter {current_iter}] Freeze {name}')
        
            #elif current_iter == self.warmup_iters + self.ft_iters:
            else:
                #runner.logger.info(f'[Iter {current_iter}] → finetune adapter')
                runner.model.module.backbone.use_selector = True
        
                #freeze selector
                for name, param in runner.model.module.backbone.named_parameters():
                    if any(key in name for key in ['selector']): 
                        param.requires_grad = False
                        #runner.logger.info(f'[Iter {current_iter}] Freeze {name}')
        
                #train adapter
                for name, param in runner.model.module.backbone.named_parameters():
                    if any(key in name for key in ['vlm_adapter', 'vfm_adapter']): 
                        param.requires_grad = True
                    #runner.logger.info(f'[Iter {current_iter}] Train {name}')            


class DualTrainingHook(Hook):
    def __init__(self, warmup_iters=6000):
        self.warmup_iters = warmup_iters

    def before_train_iter(self, runner):
        current_iter = runner.iter
        if current_iter == 0:
            runner.model.module.backbone.use_stage2 = False     
            runner.logger.info(f'[Iter {current_iter}] → Train phase1 adapter') 
            
            if runner.model.module.backbone.vlm_adapter2 is None:
                #parallel adapter, only adapter1
                #enable lr
                for e in runner.model.module.backbone.vlm_adapter1:
                    e.unfreeze_lr()
                for e in runner.model.module.backbone.vfm_adapter1:
                    e.unfreeze_lr()
                
                #disable hr
                for e in runner.model.module.backbone.vlm_adapter1:
                    e.freeze_hr()
                for e in runner.model.module.backbone.vfm_adapter1:
                    e.freeze_hr()
                runner.logger.info(f'[Iter {current_iter}] → Enable phase1 adapter, Disable phase2 adapter')
            else:
                #sequential adapter,  adapter1 and adapter2
                #enable lr
                for e in runner.model.module.backbone.vlm_adapter1:
                    e.unfreeze_lr()
                for e in runner.model.module.backbone.vfm_adapter1:
                    e.unfreeze_lr()

                #diable hr
                for e in runner.model.module.backbone.vlm_adapter2:
                    e.freeze_hr()
                for e in runner.model.module.backbone.vfm_adapter2:
                    e.freeze_hr()
                runner.logger.info(f'[Iter {current_iter}] → Enable phase1 adapter, Disable phase2 adapter')


        if current_iter == self.warmup_iters:
            runner.logger.info(f'[Iter {current_iter}] → Train phase2 adapter')
            runner.model.module.backbone.use_stage2 = True      

            if runner.model.module.backbone.vlm_adapter2 is None:
                #parallel adapter, only adapter1
                #disable lr
                for e in runner.model.module.backbone.vlm_adapter1:
                    e.freeze_lr()
                for e in runner.model.module.backbone.vfm_adapter1:
                    e.freeze_lr()
                
                #enable hr
                for e in runner.model.module.backbone.vlm_adapter1:
                    e.unfreeze_hr()
                for e in runner.model.module.backbone.vfm_adapter1:
                    e.unfreeze_hr()
                runner.logger.info(f'[Iter {current_iter}] → Disable phase1 adapter, Enable phase2 adapter')

            else:
                #sequential adapter,  adapter1 and adapter2
                #disable lr
                for e in runner.model.module.backbone.vlm_adapter1:
                    e.freeze_lr()
                for e in runner.model.module.backbone.vfm_adapter1:
                    e.freeze_lr()  
                
                #enable hr
                for e in runner.model.module.backbone.vlm_adapter2:
                    e.unfreeze_hr()
                for e in runner.model.module.backbone.vfm_adapter2:
                    e.unfreeze_hr()
                runner.logger.info(f'[Iter {current_iter}] → Disable phase1 adapter, Enable phase2 adapter')

                
class AdapterDropHook(Hook):
    def __init__(self, warmup_iters=2500, phase1=1000, phase2=2000, topk=0.25, diff_mean=0.05, min_mean=0.05):
        self.warmup_iters = warmup_iters
        self.phase1 = phase1
        self.phase2 = phase2
        #keep topk
        self.topk = round(topk*24) 
        self.diff_mean = diff_mean
        self.min_mean  = min_mean
        self.interval = self.phase2//self.topk
        self.last_drop_step = -1
    
    
    def before_train_iter(self, runner):
        current_iter = runner.iter
        #start to drop
        if current_iter > self.warmup_iters:
            #loop each adapter

            #for VFM, drop less useful adapter
            for i, adp in enumerate(runner.model.module.backbone.vlm_adapter):
                #check self
                if adp.act_self.item() and adp.running_mean['self'] < self.min_mean:
                    adp.act_self.fill_(False)
                    runner.logger.info(f'[Iter {current_iter}] Drop VLM Layer_{i}_self Due to the low gating mean')

                #check borrow
                if adp.act_borrow.item() and adp.running_mean['borrow'] < self.min_mean:
                    adp.act_borrow.fill_(False)
                    runner.logger.info(f'[Iter {current_iter}] Drop VLM Layer_{i}_borrow Due to the low gating mean')
                
                #phase1, drow less important adapter
                if current_iter < (self.warmup_iters + self.phase1):
                    if adp.act_self.item() and adp.act_borrow.item():
                        # if abs(adp.running_mean['self'] - adp.running_mean['borrow']) > self.diff_mean:
                        # Drop smaller one
                        if adp.running_mean['self'] < adp.running_mean['borrow']:
                            adp.act_self.fill_(False)
                            runner.logger.info(f'[Iter {current_iter}] Drop VLM Layer_{i}_self Due to the lower gating mean')
                        else:
                            adp.act_borrow.fill_(False)
                            runner.logger.info(f'[Iter {current_iter}] Drop VLM Layer_{i}_borrow Due to the lower gating mean')

            
            #for VLM, drop less useful adapter
            for i, adp in enumerate(runner.model.module.backbone.vfm_adapter):
                #check self
                if adp.act_self.item() and adp.running_mean['self'] < self.min_mean:
                    adp.act_self.fill_(False)
                    runner.logger.info(f'[Iter {current_iter}] Drop VFM Layer_{i}_self Due to the low gating mean')

                #check borrow
                if adp.act_borrow.item() and adp.running_mean['borrow'] < self.min_mean:
                    adp.act_borrow.fill_(False)
                    runner.logger.info(f'[Iter {current_iter}] Drop VFM Layer_{i}_borrow Due to the low gating mean')

                #phase1, drow less important adapter
                if current_iter < (self.warmup_iters + self.phase1):
                    if adp.act_self.item() and adp.act_borrow.item():
                        # if abs(adp.running_mean['self'] - adp.running_mean['borrow']) > self.diff_mean:
                        # Drop smaller one
                        if adp.running_mean['self'] < adp.running_mean['borrow']:
                            adp.act_self.fill_(False)
                        else:
                            adp.act_borrow.fill_(False)
                            

            if current_iter < (self.warmup_iters + self.phase1 + self.phase2):                    
                #second phase, drop topk         
                phase2_iter = current_iter - self.warmup_iters - self.phase1
                step_idx = phase2_iter // self.interval

                if step_idx > self.last_drop_step:
                    #drop vlm
                    layer_entropy = []
                    for i, adp in enumerate(runner.model.module.backbone.vlm_adapter):
                        if adp.act_self.item():
                            layer_entropy.append((adp.running_entropy['self'], i, 'self'))
                        if adp.act_borrow.item():
                            layer_entropy.append((adp.running_entropy['borrow'], i, 'borrow'))
                        

                    # 按 entropy 降序排序，取 top-k
                    layer_entropy.sort(reverse=True)
                    to_drop = layer_entropy[:1]

                    for _, i, key in to_drop:
                        if key == 'self':
                            runner.model.module.backbone.vlm_adapter[i].act_self.fill_(False)
                            runner.logger.info(f'[Iter {current_iter}] Drop VLM Layer_{i}_self Due to the high entropy')
                        elif key == 'borrow':
                            runner.model.module.backbone.vlm_adapter[i].act_borrow.fill_(False)
                            runner.logger.info(f'[Iter {current_iter}] Drop VLM Layer_{i}_borrow Due to the high entropy')

                    
                    #drop vfm
                    layer_entropy = []
                    for i, adp in enumerate(runner.model.module.backbone.vfm_adapter):
                        if adp.act_self.item():
                            layer_entropy.append((adp.running_entropy['self'], i, 'self'))
                        if adp.act_borrow.item():
                            layer_entropy.append((adp.running_entropy['borrow'], i, 'borrow'))
                        

                    # 按 entropy 降序排序，取 top-k
                    layer_entropy.sort(reverse=True)
                    to_drop = layer_entropy[:1]

                    for _, i, key in to_drop:
                        if key == 'self':
                            runner.model.module.backbone.vfm_adapter[i].act_self.fill_(False)
                            runner.logger.info(f'[Iter {current_iter}] Drop VFM Layer_{i}_self Due to the high entropy')
                        elif key == 'borrow':
                            runner.model.module.backbone.vfm_adapter[i].act_borrow.fill_(False)
                            runner.logger.info(f'[Iter {current_iter}] Drop VFM Layer_{i}_borrow Due to the high entropy')

                    
                    self.last_drop_step = step_idx



                

# class LogWeightHook(Hook):
#     def __init__(self, json_path, mini_interval, log_interval):
#         self.mini_interval = mini_interval
#         self.log_interval = log_interval
#         self.jsonl_path = json_path

#         # with open(self.jsonl_path, 'w') as f:
#         #     f.write('')
        
#     def after_train_iter(self, runner):
#         current_iter = runner.iter
#         mymodel = runner.model.module

#         if current_iter % self.mini_interval == 0:
#             vlm_mean_std = {}
#             for lid, w in mymodel.w_vlm.items():
#                 vlm_mean_std[str(lid)] = {
#                     'mean': w.mean().item(),
#                     'std': w.std().item()
#                 }
    
#             vfm_mean_std = {}
#             for lid, w in mymodel.w_vfm.items():
#                 vfm_mean_std[str(lid)] = {
#                     'mean': w.mean().item(),
#                     'std': w.std().item()
#                 }
    
                    
#             mean_std_record = {
#                 'iter': current_iter,
#                 'type': 'mean_std',
#                 'vlm': vlm_mean_std,
#                 'vfm': vfm_mean_std,
#             }
    
#             with open(self.jsonl_path, 'a') as f:
#                 f.write(json.dumps(mean_std_record) + '\n')

#         # --------- Log full w at interval ----------
#         if current_iter % self.log_interval == 0:
#             vlm_full = {}
#             for lid, w in mymodel.w_vlm.items():
#                 vlm_full[str(lid)] = w.detach().cpu().numpy().tolist()

#             vfm_full = {}
#             for lid, w in mymodel.w_vfm.items():
#                 vfm_full[str(lid)] = w.detach().cpu().numpy().tolist()

#             full_record = {
#                 'iter': current_iter,
#                 'type': 'full_w',
#                 'vlm': vlm_full,
#                 'vfm': vfm_full,
#             }

#             with open(self.jsonl_path, 'a') as f:
#                 f.write(json.dumps(full_record) + '\n')

class LogWeightHook(Hook):
    def __init__(self, json_path, mini_interval, log_interval):
        self.mini_interval = mini_interval
        self.log_interval = log_interval
        self.jsonl_path = json_path

        # with open(self.jsonl_path, 'w') as f:
        #     f.write('')
        
    def after_train_iter(self, runner):
        current_iter = runner.iter
        mymodel = runner.model.module

        if current_iter % self.mini_interval == 0:
            s_vlm_mean_std = {}
            for lid, w in mymodel.ws_vlm.items():
                s_vlm_mean_std[str(lid)] = {
                    'mean': w.mean().item(),
                    'std': w.std().item()
                }
    
            s_vfm_mean_std = {}
            for lid, w in mymodel.ws_vfm.items():
                s_vfm_mean_std[str(lid)] = {
                    'mean': w.mean().item(),
                    'std': w.std().item()
                }

            b_vlm_mean_std = {}
            for lid, w in mymodel.wb_vlm.items():
                b_vlm_mean_std[str(lid)] = {
                    'mean': w.mean().item(),
                    'std': w.std().item()
                }
    
            b_vfm_mean_std = {}
            for lid, w in mymodel.wb_vfm.items():
                b_vfm_mean_std[str(lid)] = {
                    'mean': w.mean().item(),
                    'std': w.std().item()
                }
    
                    
            mean_std_record = {
                'iter': current_iter,
                'type': 'mean_std',
                'vlms': s_vlm_mean_std,
                'vfms': s_vfm_mean_std,
                'vlmb': b_vlm_mean_std,
                'vfmb': b_vfm_mean_std,
            }
    
            with open(self.jsonl_path, 'a') as f:
                f.write(json.dumps(mean_std_record) + '\n')
