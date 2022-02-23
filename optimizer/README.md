# optimizer
[GitHub](https://viblo.asia/p/optimizer-hieu-sau-ve-cac-thuat-toan-toi-uu-gdsgdadam-Qbq5QQ9E5D8)

## 1. Optimizer
* Optimizer là cơ sở để xây dựng mô hình neural network với mục đích "học " được các features ( hay pattern) của dữ liệu đầu vào, từ đó có thể tìm 1 cặp weights và bias phù hợp để tối ưu hóa model.
* Vai trò:  Giúp cải thiện weight và bias.

## 2. Các Optimizer Phổ biến.
![optimizer](https://user-images.githubusercontent.com/86842861/126857441-c00c5927-1db4-4820-9f2d-d8ce5c75da3a.png)


### 2.1. Gradient Descent

#### Định nghĩa
* Gradient Descent (giảm dần độ dốc): là chọn 1 nghiệm ngẫu nhiên cứ sau mỗi vòng lặp (hay epoch) thì cho nó tiến dần đến điểm local.
* Công thức : xnew = xold - alpha * d(xold)
* Hypepatameter: alpha
#### Vấn đề xảy ra khi sử dụng GD? 
* Quá trình hội tụ phụ thuộc vào điểm khởi tạo

![SGD1](https://user-images.githubusercontent.com/86842861/126857450-6adf236e-5e47-40ae-9664-0cbcf72d8d31.gif)
![SGD](https://user-images.githubusercontent.com/86842861/126857453-3bfee6fd-c544-47b5-9341-45d00b87460d.gif)

* hoặc tốc độ học (learning rate) quá lớn hoặc quá nhỏ cũng ảnh hưởng: nếu tốc độ học quá nhỏ thì tốc độ hội tụ rất chậm ảnh hưởng đến quá trình training, còn tốc độ học quá lớn thì tiến nhanh tới đích sau vài vòng lặp tuy nhiên thuật toán không hội tụ, quanh quẩn quanh đích vì bước nhảy quá lớn.

![SG](https://user-images.githubusercontent.com/86842861/126857458-395f4991-b429-4548-8329-5ee3235e4ba7.gif)
![SGD4](https://user-images.githubusercontent.com/86842861/126857460-6194a78d-fc60-4b79-83df-408496de5277.gif)

#### Ưu điểm
* Thuật toán gradient descent cơ bản, dễ hiểu. 
* Thuật toán đã giải quyết được vấn đề tối ưu model neural network bằng cách cập nhật trọng số sau mỗi vòng lặp.

#### Nhược điểm
* Vì đơn giản nên thuật toán Gradient Descent còn nhiều hạn chế như phụ thuộc vào nghiệm khởi tạo ban đầu và learning rate.
* Ví dụ 1 hàm số có 2 global minimum thì tùy thuộc vào 2 điểm khởi tạo ban đầu sẽ cho ra 2 nghiệm cuối cùng khác nhau.
* Tốc độ học quá lớn sẽ khiến cho thuật toán không hội tụ, quanh quẩn bên đích vì bước nhảy quá lớn; hoặc tốc độ học nhỏ ảnh hưởng đến tốc độ training.

### 2.2 Stochastic Gradient Descent (SGD)
#### Định nghĩa
* Stochastic Gradient Descent (SGD) là 1 biến thể của Gradient Descent . Thay vì sau mỗi epoch chúng ta sẽ cập nhật trọng số (Weight) 1 lần thì trong mỗi epoch có N điểm dữ liệu chúng ta sẽ cập nhật trọng số N lần. 
* Nhìn vào 1 mặt, SGD sẽ làm giảm đi tốc độ của 1 epoch. Tuy nhiên nhìn theo 1 hướng khác, SGD sẽ hội tụ rất nhanh chỉ sau vài epoch. Công thức SGD cũng tương tự như GD nhưng thực hiện trên từng điểm dữ liệu.
![SGD3](https://user-images.githubusercontent.com/86842861/126857609-76d99ddd-ae5a-43fe-b919-7e4c71055a03.png)

* Công thức : xnew = xold - alpha * d(xold)
* Hypepatameter: alpha

#### Ưu điểm
* Thuật toán giải quyết được đối với cơ sở dữ liệu lớn mà GD không làm được.

#### Nhược điểm
* Thuật toán vẫn chưa giải quyết được 2 nhược điểm lớn của gradient descent (learning rate và điểm dữ liệu ban đầu). Vì vậy ta phải kết hợp SGD với 1 số thuật toán khác như: Momentum, AdaGrad,..

### 2.3. Gradient Descent with Momentum
* Để khắc phục các hạn chế trên của thuật toán Gradient Descent người ta dùng Gradient Descent with Momentum
![momentum](https://user-images.githubusercontent.com/86842861/126857768-52b360b4-c356-4f8b-b887-a35c4fdcbf18.png)


#### Định nghĩa
* Công thức: 
```V = beta * V(old) + (1 - beta)dX; X(new) = X(old) - alpha * V```
* Hypepatameters: alpha, beta=0.9
* GD without Momentum
![without mmt](https://user-images.githubusercontent.com/86842861/126857749-a1721f96-6521-4e5f-89ef-cd5e3e00e437.gif)

* GD with Momentum
![with mmt](https://user-images.githubusercontent.com/86842861/126857762-5618073a-26ca-471a-b8d9-76bdbca34456.gif)

#### Ưu điểm :
* Thuật toán tối ưu giải quyết được vấn đề: Gradient Descent không tiến được tới điểm global minimum mà chỉ dừng lại ở local minimum.
* Tiến tới khu vực local nhanh hơn (có vận tốc)
#### Nhược điểm :
* Tuy momentum giúp hòn bi vượt dốc tiến tới điểm đích, tuy nhiên khi tới gần đích, nó vẫn mất khá nhiều thời gian giao động qua lại trước khi dừng hẳn, điều này được giải thích vì viên bi có đà.

### 2.4 Adagrad
#### Định nghĩa
* Không giống như các thuật toán trước đó thì learning rate hầu như giống nhau trong quá trình training (learning rate là hằng số).
* Adagrad coi learning rate là 1 tham số. Tức là Adagrad sẽ cho learning rate biến thiên sau mỗi thời điểm t.
* Công thức:

![ct ](https://user-images.githubusercontent.com/86842861/126857843-581f0db0-2d87-4876-bfaa-2341b6fdf4b4.png)
```
Trong đó :
n : hằng số
gt : gradient tại thời điểm t
ϵ : hệ số tránh lỗi ( chia cho mẫu bằng 0)
G : là ma trận chéo mà mỗi phần tử trên đường chéo (i,i) là bình phương của đạo hàm vectơ tham số tại thời điểm t.
```

#### Ưu điểm :
* Một lơi ích dễ thấy của Adagrad là tránh việc điều chỉnh learning rate bằng tay, chỉ cần để tốc độ học default là 0.01 thì thuật toán sẽ tự động điều chỉnh.
#### Nhược điểm :
* Yếu điểm của Adagrad là tổng bình phương biến thiên sẽ lớn dần theo thời gian cho đến khi nó làm tốc độ học cực kì nhỏ, làm việc training trở nên đóng băng.

### 2.5 RMS prop
#### Định nghĩa
* RMSprop giải quyết vấn đề tỷ lệ học giảm dần của Adagrad bằng cách chia tỷ lệ học cho trung bình của bình phương gradient.
* Công thức:
![RMSprop](https://user-images.githubusercontent.com/86842861/126862320-6849bfb8-036e-45e8-aaeb-87a277c0a7c3.png)

*Hypeparameters: n (alpha), beta=0.9 (1 - beta = 0.1)
#### Ưu điểm :
* Ưu điểm rõ nhất của RMSprop là giải quyết được vấn đề tốc độ học giảm dần của Adagrad ( vấn đề tốc độ học giảm dần theo thời gian sẽ khiến việc training chậm dần, có thể dẫn tới bị đóng băng )
#### Nhược điểm :
* Thuật toán RMSprop có thể cho kết quả nghiệm chỉ là local minimum chứ không đạt được global minimum như Momentum. Vì vậy người ta sẽ kết hợp cả 2 thuật toán Momentum với RMSprop cho ra 1 thuật toán tối ưu Adam. Chúng ta sẽ trình bày nó trong phần sau.




### 2.6 Adam
#### Định nghĩa
* Adam là sự kết hợp của Momentum và RMSprop. 
* Nếu giải thích theo hiện tượng vật lí thì Momentum giống như 1 quả cầu lao xuống dốc, còn Adam như 1 quả cầu rất nặng có ma sát, vì vậy nó dễ dàng vượt qua local minimum tới global minimum và khi tới global minimum nó không mất nhiều thời gian dao động qua lại quanh đích vì nó có ma sát nên dễ dừng lại hơn.
![Adam](https://user-images.githubusercontent.com/86842861/126862397-7328dddf-9630-4ce5-b382-b1d7c74385cd.png)
* Công thức:
![CT Adam](https://user-images.githubusercontent.com/86842861/126862417-dcab09d9-d2be-4168-80a0-8171803b0a70.png)

* Hypeparameters: alpha, beta_1, beta_2

#### Ưu điểm:
* Là phương pháp giúp tối ưu giữa RMSprop và SGD with Momentum: Giúp chạm tới điểm glocal minimun mà không mất nhiều thời gian gian động qua lại trước khi tới điểm glocal.

#### Nhược điểm
* Phức tạp, có nhiều hypeparameters hơn các phương pháp còn lại.

## 3. Test with Hymenoptera dataset (model: mobilenet_v2)
### 3.1 SGD
#### Have scheduler
* optimizer = optim.SGD(model.parameters(), lr=0.01)
* Time train (30 epochs): 1m 53s
* Best Validation Accuracy: 91.833333

#### Not scheduler
* optimizer = optim.SGD(model.parameters(), lr=0.01)
* Time train (30 epochs): 1m 54s
* Best Validation Accuracy: 92.645833

### 3.2 SGD with Momentum
#### Have scheduler
* optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
* Time train (30 epochs): 1m 55s
* Best Validation Accuracy: 94.208333

#### Not scheduler
* optimizer = optim.SGD(model.parameters(), lr=0.01)
* Time train (30 epochs): 2m 54s
* Best Validation Accuracy: 93.687500

### 3.3 Adagrad
#### Have scheduler
* optimizer = optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-8)
* Time train (30 epochs): 1m 52s
* Best Validation Accuracy: 93.979167

#### Not scheduler
* optimizer = optim.SGD(model.parameters(), lr=0.01)
* Time train (30 epochs): 2m 51s
* Best Validation Accuracy: 93.979167

### 3.4 RMSprop
##### Have scheduler
* optimizer = optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-8)
* Time train (30 epochs): 1m 53s
* Best Validation Accuracy: 93.979167

##### Not scheduler
* optimizer = optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-8)
* Time train (30 epochs): 1m 54s
* Best Validation Accuracy: 95.020833

### 3.5 Adam
#### Have scheduler
* optimizer = optim.Adam(params=model.parameters(), lr=0.01, betas=(0.9, 0.999),
					   eps=1e-08, weight_decay=0, amsgrad=False)
* Time train (30 epochs): 1m 51s
* Best Validation Accuracy: 95.541667

#### Not scheduler
* optimizer = optim.Adam(params=model.parameters(), lr=0.01, betas=(0.9, 0.999),
					   eps=1e-08, weight_decay=0, amsgrad=False)
* Time train (30 epochs): 1m 53s
* Best Validation Accuracy: 96.062500

### Note:
* Thời gian training giữa các thuật toán tối ưu chênh lệch không nhiều trong đó SGD và Adam thấp nhất (1m 51s trong 30 epochs)
* Highest accuracy: Adam - 96.062500

Mới train 1 lần nên kết quả cũng chưa khách quan. Nhưng theo lý thuyết thì Adam là phương pháp optimizer tối ưu nhất (kết hợp của SGD momentum với RMSprop - 2 phương pháp tối ưu khá tốt).

## 4. Maxima, Minima, Saddle point
![minmaxsaddle](https://user-images.githubusercontent.com/86842861/127491609-30293666-bf25-42a1-b3f8-19da96b73c75.png)

### 4.1 Maxima, Minima
[GitHub](https://www.youtube.com/watch?v=ux7EQ3ip2DU)

### 4.2 Saddle point
[GitHub](https://www.youtube.com/watch?v=8aAU4r_pUUU)
#### Khái niệm
* Saddle point - điểm yên ngựa hoặc điểm minimax là một điểm trên bề mặt đồ thị của hàm trong đó các sườn theo hướng trực giao đều bằng 0, nhưng không phải là điểm cực trị cục bộ của hàm.
![Saddle_point svg](https://user-images.githubusercontent.com/86842861/127491625-a8854a2a-fef3-4531-a44c-35a562fd8f0a.png)


https://www.khanacademy.org/math/multivariable-calculus/applications-of-multivariable-derivatives/optimizing-multivariable-functions/a/maximums-minimums-and-saddle-points
* Saddle points: là những điểm ổn định trong đó hàm có cực đại cục bộ theo một hướng, nhưng cực tiểu cục bộ theo hướng khác.

#### Escaping from Saddle points
https://www.offconvex.org/2016/03/22/saddlepoints/
![escapesmall](https://user-images.githubusercontent.com/86842861/127491642-a3bac6ce-f6f3-4d71-9b97-03866b2e5b9f.png)

https://www.youtube.com/watch?v=p3zZTthoLIQ (28:36)





