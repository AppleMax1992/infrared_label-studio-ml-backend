通过在标注工具label-studio中集成yolo-seg的方法实现 根据少量标注数据（样本10-50组），对未标注数据预打标，提升标注效率。并且后续标注人员可通过label-studio对预标注进行调整，简化标注流程。
# 1. 部署label-studio （目的是启动UI页面与标注人员交互）
## 1.1 安装所需环境
(由于label-studio所需依赖较多，选择用poetry安装)
### 安装poetry
pip install poetry
(poetry 创建新环境)
poetry new my-label-studio
cd my-label-studio
在poetry中安装label-studio环境
poetry add label-studio
### activate poetry environment
poetry shell

## 1.2 安装label-studio
### 源代码安装(Optional 推荐：后续可根据需要自定义开发)
https://github.com/HumanSignal/label-studio

### 映射静态文件的路径 (Optional 推荐：不做可能会出现图片不能正常显示的情况)
python label_studio/manage.py migrate
python label_studio/manage.py collectstatic

### pip 直接安装(Optional 不需要开发)
pip install -U label-studio

## 1.3 本地启动label-studio 服务 
(以autodl部署为例，指定端口为6006。本地服务器可以自定义端口)
### 参数解释 port(自定义的端口) data-dir 自定义文件存储路径，包含文件上传存储路径和标注之后的文件存储路径。
label-studio start --port 6006 --data-dir mydata

# 2. 部署label-studio-ml-backend （目的自定义的yolo-seg模型）
## 安装label-studio-ml-backend
git clone https://github.com/HumanSignal/label-studio-ml-backend
cd label-studio-ml-backend/
pip install -e .
### 创建自定义后端ML组件 
label-studio-ml create my_ml_backend
## 启动 组件
label-studio-ml start my_ml_backend 
### 配置服务端口(Optional 推荐：由于后续要通过端口集成到label-studio中，因此设定端口回避端口被占有无法相应请求的情况)
label-studio-ml start my_ml_backend --port 9090

# 3. 集成label-studio-ml-backend 服务到label-studio主程序中
## 浏览器访问label-studio 
浏览器地址 http://0.0.0.0:6006/（6006为1.3自定义端口）
点Setting如图1所示
![image](https://github.com/AppleMax1992/label-studio--/assets/77500295/8842ec18-a909-4db7-b977-330e4b6e448b)图1
选择Model
![image](https://github.com/AppleMax1992/label-studio--/assets/77500295/1e658630-3b00-4b2e-9d87-26a38ab6985b)
配置label-studio-ml-backend服务的地址
![image](https://github.com/AppleMax1992/label-studio--/assets/77500295/22306138-8aef-4208-bec8-59c2c01b0708)
保存之后可以看到服务已经成功链接
![image](https://github.com/AppleMax1992/label-studio--/assets/77500295/c2f683e5-7c02-4ff6-93b9-10d54f75598d)




## 在label-studio页面中上传图片
上传新图片
![image](https://github.com/AppleMax1992/label-studio--/assets/77500295/a98d04b7-73b5-4c40-856b-e2e0152f700b)
![image](https://github.com/AppleMax1992/label-studio--/assets/77500295/40599056-89fa-4c93-811e-14a051519112)
从后端服务已经成功log出预测结果
![image](https://github.com/AppleMax1992/label-studio--/assets/77500295/a32dfeb0-bfbc-4833-8081-a8e4d71e66f7)

点击新增图片的show task socre
![image](https://github.com/AppleMax1992/label-studio--/assets/77500295/8196ad63-bf91-4c4b-8d67-c2f6137bb2c3)


可以看到label-studio中已经生成了可以编辑的目标框
![image](https://github.com/AppleMax1992/infrared_label-studio-ml-backend/assets/77500295/bbdf2096-8c10-4d3a-a2b3-1a895eebe7b6)




# 4. TODO 在图片编辑页面展示预测出的标签和mask，后续标注人员可以以审核的方式，对图片进行修改，无需全部点位手动标注，简化标注流程


