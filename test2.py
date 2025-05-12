for epoch in range(start_epoch, config['training']['num_epochs']):
    logger.info(f"--- Starting Epoch {epoch}/{config['training']['num_epochs']} ---")
    epoch_loss = 0.0
    steps_in_epoch = 0

    for step, batch in enumerate(train_dataloader):
        # updata_back = False #不再需要这个单独的标志

        preprocessed_batch = model.data_preprocess(batch)
        with accelerator.accumulate(model):
            outputs = model(preprocessed_batch)
            loss = outputs.get("loss")
            print('loss:', loss)

            # --- 为 backward() 准备 loss，并为 gather() 准备原始 loss ---
            # 克隆原始 loss 用于 gather 和 avg_loss 计算
            original_loss_for_gathering = loss.detach().clone()
            # 这个 loss 将用于 backward，如果原始 loss 是 NaN/Inf，它将被替换
            loss_for_backward = loss

            if loss is None:  # 最好也处理 None 的情况
                logger.error(
                    f"Rank {accelerator.process_index} - Step {global_step}: Loss is None! This is unexpected. "
                    f"Using zero loss for backward pass to maintain DDP synchronization."
                )
                # 对于 backward，使用一个需要梯度的零张量
                loss_for_backward = torch.tensor(0.0, device=accelerator.device, dtype=torch.float32,
                                                 requires_grad=True)  # 确保dtype匹配
                # 对于 gathering，确保它是一个 NaN，以便 avg_loss 正确处理
                original_loss_for_gathering = torch.tensor(float('nan'), device=accelerator.device, dtype=torch.float32)

            elif torch.isnan(loss).any() or torch.isinf(loss).any():
                logger.warning(
                    f"Rank {accelerator.process_index} - Step {global_step}: Local loss is {loss.item()}. "
                    f"Using zero loss for this rank's backward pass to maintain DDP synchronization."
                )
                # 对于 backward，使用一个需要梯度的零张量
                loss_for_backward = torch.tensor(0.0, device=loss.device, dtype=loss.dtype, requires_grad=True)
                # original_loss_for_gathering 已经包含了 NaN/Inf 值，无需更改

            # --- NaN 处理逻辑开始 (使用 original_loss_for_gathering) ---
            gathered_losses = accelerator.gather(original_loss_for_gathering)
            is_nan_or_inf_mask = torch.isnan(gathered_losses) | torch.isinf(gathered_losses)  # 同时检查 NaN 和 Inf
            valid_losses = gathered_losses[~is_nan_or_inf_mask]

            if valid_losses.numel() > 0:
                avg_loss = valid_losses.mean()
            else:
                avg_loss = torch.tensor(float('nan'), device=gathered_losses.device,
                                        dtype=gathered_losses.dtype)
                logger.warning(f"Step {global_step}: All gathered losses were NaN/Inf. avg_loss set to NaN.")
            # --- NaN 处理逻辑结束 ---

            if not (torch.isnan(avg_loss) or torch.isinf(avg_loss)):  # 检查 avg_loss
                epoch_loss += avg_loss.item() / accelerator.gradient_accumulation_steps
            else:
                logger.warning(f"Step {global_step}: avg_loss is NaN/Inf. Skipping accumulation into epoch_loss.")

            steps_in_epoch += 1
            # avg_loss2 的计算也应该基于有效的 gathered_losses 或者直接使用 avg_loss
            # 确保这里的 gather 不会因为某个 rank 的 loss_for_backward 是0而产生误导性结果
            # 应该 gather original_loss_for_gathering
            gathered_for_avg_loss2 = accelerator.gather(
                original_loss_for_gathering.repeat(config['dataloader']['batch_size_per_device']))
            valid_for_avg_loss2 = gathered_for_avg_loss2[
                ~(torch.isnan(gathered_for_avg_loss2) | torch.isinf(gathered_for_avg_loss2))]
            if valid_for_avg_loss2.numel() > 0:
                avg_loss2 = valid_for_avg_loss2.mean()
            else:
                avg_loss2 = torch.tensor(float('nan'), device=original_loss_for_gathering.device,
                                         dtype=original_loss_for_gathering.dtype)

            # --- 反向传递 ---
            # 所有 rank 都执行 backward。
            # 如果原始 loss 是 NaN/Inf，则 loss_for_backward 是一个零张量，其梯度为零。
            accelerator.backward(loss_for_backward)

            if accelerator.sync_gradients:
                # 梯度会在这里同步。有 NaN/Inf 原始 loss 的 rank 会贡献零梯度。
                # 这不会破坏同步过程。
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                progress_bar.update(1)

                # --- Logging ---
                # 确保在记录时，avg_loss 和 avg_loss2 不是 NaN/Inf，或者处理这种情况
                log_avg_loss = avg_loss.item() if not (torch.isnan(avg_loss) or torch.isinf(avg_loss)) else float('nan')
                log_avg_loss2 = avg_loss2.item() if not (torch.isnan(avg_loss2) or torch.isinf(avg_loss2)) else float(
                    'nan')

                if global_step % 5 == 1:
                    lr = optimizer.param_groups[0]['lr']
                    logger.info(
                        f"Epoch {epoch} | Step {global_step}/{max_train_steps} | Loss: {log_avg_loss:.4f}/{log_avg_loss2:.4f} | LR: {lr:.2e}")
                    if tracker:
                        if not (torch.isnan(avg_loss) or torch.isinf(avg_loss)):  # 只记录有效的avg_loss
                            accelerator.log({
                                "train/loss": avg_loss.item(),
                                "train/learning_rate": lr,
                            }, step=global_step)
                        else:
                            logger.warning(f"Step {global_step}: avg_loss is NaN/Inf, not logging to tracker.")
                # ... (其余代码保持不变) ...