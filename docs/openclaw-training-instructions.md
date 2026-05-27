# OpenClaw 远程训练执行指令

在 Windows 笔记本（RTX 4060）上训练医疗实体匹配模型。

## 前置信息
- GitHub 仓库: `https://github.com/Bensonluo/4bit-QLoRA-post-training.git`
- Windows SSH: `sshuser@ben-workstation.local`，密码 `sbglm5`
- Windows 项目路径: `C:\Users\sshuser\4bit-QLoRA-post-training\`
- Windows 使用 PowerShell（不能用 `&&`，用分号 `;`）
- 训练预计耗时: 3-5 小时

---

## 第一步：Clone 代码（首次）或拉取更新

```bash
# 如果目录不存在
git clone https://github.com/Bensonluo/4bit-QLoRA-post-training.git ~/Documents/GitHub/4bit-QLoRA-post-training

# 如果已经 clone 过
cd ~/Documents/GitHub/4bit-QLoRA-post-training && git pull
```

---

## 第二步：打包项目

```bash
tar cf /tmp/project.tar --exclude='.git' --exclude='venv' --exclude='.venv' --exclude='outputs' --exclude='__pycache__' --exclude='.omc' --exclude='.claude' --exclude='*.egg-info' --exclude='data/external' -C ~/Documents/GitHub/4bit-QLoRA-post-training .
```

---

## 第三步：传输到 Windows

```bash
scp /tmp/project.tar sshuser@ben-workstation.local:C:/Users/sshuser/
```

密码: `sbglm5`

---

## 第四步：SSH 到 Windows 解压并启动训练

```bash
ssh sshuser@ben-workstation.local
```

然后在 Windows 上执行：

```powershell
cd C:\Users\sshuser\4bit-QLoRA-post-training
tar xf C:\Users\sshuser\project.tar
.\venv\Scripts\python.exe -u scripts/train_medical_entity.py --mac-1b --epochs 1 --output-dir ./outputs/medical-entity-win-1b-v2
```

训练过程中会显示 loss 和进度，等待完成不要中断。

---

## 第五步：训练完成后复制 checkpoint 回 Mac

```bash
mkdir -p ~/Documents/GitHub/4bit-QLoRA-post-training/outputs/medical-entity-win-1b-v2
scp -r sshuser@ben-workstation.local:C:/Users/sshuser/4bit-QLoRA-post-training/outputs/medical-entity-win-1b-v2/checkpoint-*/ ~/Documents/GitHub/4bit-QLoRA-post-training/outputs/medical-entity-win-1b-v2/
```

---

## 第六步：报告结果

1. 最终 loss 值
2. 训练总步数
3. checkpoint 路径
4. 训练耗时

---

## 故障排查

- SSH 不通：检查 Windows 是否开机、同网络、OpenSSH 服务是否运行
- CUDA 不可用：Windows 上执行 `nvidia-smi` 检查驱动
- OOM 显存不足：训练命令加 `--batch-size 1`
- 训练中断：重跑相同命令，会自动从最后一个 checkpoint 恢复
