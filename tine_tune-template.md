1、
    定义TrainingArguments和DatasetArguments；（TrainingArguments在transformers中有默认值）
    定义数据集处理器；

    模型参数包括：
        model_name_or_path：模型的名称或路径


2、
    HfArgumentParser解析TrainingArguments、DatasetArguments中定义的参数；

3、
    加载模型：
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            script_args.model_name_or_path,     # 模型路径
            trust_remote_code=True,             # 如果设置为 True，表示在本地加载模型时会跳过远程代码的校验，可以提高模型加载速度。
            torch_dtype='auto',                 # 字符串类型的参数，指定pytorch张量的数据类型，auto表示自动选择合适的数据类型
            device_map='auto'                   # model使用的设备，auto表示自动选择
        )
        print(model.config)    # 可以修改模型参数

        # 添加lora配置，
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=32,                                       # lora的秩，即矩阵分解的维度
            # lora_alpha=LORA_ALPHA,                        
            target_modules=["o_proj", "W_pack",],       # 需要lora微调哪些线性层，通常是输出层和输入层的权重矩阵
            lora_dropout=0.05,                          # dropout
            bias="none",            # Lora 模型中是否考虑偏置项，可选值为 'none'、'factorized' 和 'full'
            task_type="CAUSAL_LM",      # Lora 模型微调的任务类型，通常是文本生成任务（例如 CausalLM）或文本分类任务（例如 SequenceClassification）
        )
        model = get_peft_model(model, lora_config)      # 根据原模型和lora参数，加载peft模型


        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            script_args.model_name_or_path,             # 模型保存地址
            trust_remote_code=True                      # 本地模型需要增加这个参数
        )
        print(tokenizer)    # 修改tokenizer的配置


        # 训练数据、测试数据处理
        pass


        # 加载训练器
        trainer = Trainer(
            model=model,                    # 待训练模型
            args=training_args,             # 训练参数
            train_dataset=dataset,          # 熟练数据
        )
        # 自定义MyTrainer
        trainer = MyTrainer(
            model=model,                        # 待训练模型
            args=training_args,                 # 训练参数
            train_dataset=dataset,              # 训练数据，通常是Dataset类
            data_collator=GroupCollator(),      # 一个自定义的数据合并器
        )
        # dpo
        dpo_trainer = DPOTrainer(
            model,                                                      # 训练的模型。
            model_ref,                                                  # 策略梯度算法中的参考模型，对比采样模型和参考模型的差异来计算梯度。
            args=training_args,                                         # 训练的参数，如batch size、learning rate等。
            beta=script_args.beta,                                      # DPO算法中的超参数，控制采样模型和参考模型的权重。
            train_dataset=train_dataset,                                # 训练数据集。
            eval_dataset=eval_dataset,                                  # 验证数据集。
            tokenizer=tokenizer,                                        # 分词器，将文本转化为模型可接受的输入格式。
            max_length=script_args.max_length,                          # 输入文本的最大长度。
            max_target_length=script_args.max_target_length,            # 目标文本的最大长度。
            max_prompt_length=script_args.max_prompt_length,            # 前缀文本的最大长度。
            generate_during_eval=True,                                  # 是否在验证时生成数据，用于计算BLEU分数等指标。
        )


        # 开始训练
        dpo_trainer.train()
