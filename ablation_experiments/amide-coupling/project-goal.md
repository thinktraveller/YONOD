# project-goal

这是一个用来验证对于morgan描述符而言，向量的拼接顺序是否对结果有所影响的项目"different_order"

## 任务目标

描述符是将一个SMILES结构式转换为若干维度向量后进行机器学习的工具，对于多个分子参与建模时，一种常见的方法是将多个分子向量化后按一定顺序拼接；

以下为建模时会用到的列：

  [1] reactant-amide
  [2] reactant-acid
  [3] product
  [4] activation
  [5] additive
  [6] base
  [7] solvent

为了方便表示，直接使用列序号的组合作为任务名称

如任务名为"012345"，表示参与建模向量化的列序号为0、1、2、3、4、5 ，且分子指纹向量的拼接顺序为0-1-2-3-4-5；

为此，我们需要测试以下任务：

12
123
124
125
126
127
1234
1235
1236
1237
1245
1246
1247
1256
1257
1267
12345
12346
12347
12356
12357
12367
12456
12457
12467
12567
123456
123457
123467
123567
124567
1234567

## 执行方案

### 1、创建新文件夹

在 different_order 下按任务名称（如"12"）创建文件夹，一共需要创建31个文件夹

### 2、批量复制与重命名yonod_config.json文件

在 different_order/1234567 中存放着初始的json文件，different_order/1234567/1234567_yonod_config.json

将其复制到每个项目文件夹中，并将文件名的开头修改为对应的任务名，如放置在 different_order/12 下的json应重命名为 different_order/12/12_yonod_config.json

### 3、批量修改json文件

首先，将json中的 "project_name" 字段修改为与任务名称一致；

如放置在 different_order/01 下的json应将 "project_name" 从 "1234567" 改成 "12" ；

然后，将 "descriptors" 字段下每个描述符的 "columns" 字段按实际参与建模的列进行对应的修改；

如初始的任务名称为 "1234567" ，代表列以1-2-3-4-5-6-7的方式建模，所以对应的 columns 写为：

```json
    {
      "descriptor": "maf",
      "mode": "sum",
      "columns": [
        "reactant-amide",
        "reactant-acid",
        "product",
        "activation",
        "additive",
        "base",
        "solvent"
      ]
    }
```

那么对于任务名称为 "12" 的任务，对应的 columns 应写为：

```json
    {
      "descriptor": "maf",
      "mode": "sum",
      "columns": [
        "reactant-amide",
        "reactant-acid"
      ]
    }
```

注意，**所有描述符都需要进行修改**；

### 4、生成脚本用来批量生成建模命令

以任务名称为 "12" 的任务为例，启动命令如下：

```bash
python main.py --config "different_order/12/12_yonod_config.json" --csv "different_order/amide-coupling(additive_fixed)_normalized_dataset"
```

其中，"--config" 指定了json文件的位置，"--csv"则指定了标准化数据集的位置，不需要改变；

生成一个python脚本用来批量执行31个任务，交付该脚本，并且给出后台执行该脚本的命令示例；

