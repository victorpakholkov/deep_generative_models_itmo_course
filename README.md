**Пахолков Виктор Владимирович**

**Курс "Глубокие генеративные модели (Deep Generative Models)" в AITalantedHub**


## ДЗ 2. Имплементация GAN

### 0. Data

Для выполнения данной работы был использован датасет CelebA.
К сожалению, через pytorch выгрузить его не получилось, поэтому он был выгружен вручную. 

Далее датасет был преобразован для более удобной работы, он был кропнут и ресайзнут (к 128), нормализован и обернут даталоадером с размером батча 64.

### 1. CSPup блок

CSPup блок был имплементирован следующим образом:

```
class CSPup(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.kernel_size =kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.split_channels = self.in_channels//2
        self.left = nn.Sequential(
            nn.ConvTranspose2d(self.split_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=False),
        )
        self.right = nn.Sequential(
            nn.Conv2d(self.split_channels, self.out_channels , 1, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose2d(self.out_channels, self.out_channels , self.kernel_size, self.stride, self.padding, bias=False),
            nn.Conv2d(self.out_channels, self.out_channels , 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.out_channels, 1, 1, 0),
        )
    def forward(self, x):
        x1 = x[:, :self.split_channels, ...]
        x2 = x[:, self.split_channels:, ...]
        y = self.left(x1)+self.right(x2)
        return y
```

Входные данные разделяются на две части: x1 и x2. x1 состоит из первой половины каналов входных данных, а x2 - из второй половины. Затем к x1 применяется последовательность слоев, определенная в self.left, которая состоит из одного слоя ConvTranspose2d. К x2 применяется последовательность слоев, определенная в self.right, которая состоит из слоя Conv2d, за которым следует слой ReLU, слой ConvTranspose2d, еще один слой Conv2d, еще один слой ReLU и заключительный слой Conv2d.

Затем результаты применения этих последовательностей слоев к x1 и x2 суммируются, чтобы получить выходной тензор y.

Этот подход позволяет уменьшить вычислительную сложность, поскольку меньшее количество каналов обрабатывается в последовательности слоев self.right. Кроме того, разделение входных данных на две части и применение к ним разных преобразований может помочь улучшить точность модели, поскольку она может учитывать разные особенности входных данных.

### 2. GAN

После имплементации CSPup был имплементирован генератор GAN по заданной в репозитории с дз архитектурной схеме.

Он состоит из генератора следующего типа:

```
class Generator(nn.Module):
    def __init__(self, ngpu):
        super().__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            CSPup(nz, ngf * 16, 4, 1, 0),
            CSPup(ngf * 16, ngf * 8, 4, 2, 1),
            CSPup(ngf * 8, ngf * 4, 4, 2, 1),
            CSPup(ngf * 4, ngf * 2, 4, 2, 1),
            CSPup(ngf * 2, ngf, 4, 2, 1),
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)
```

И дискриминатора следующего типа:

```
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
```

Сама сеть инициализируется с помощью функции initialize(), в которую может подаваться количество доступных гпу:

```
def initialize(ngpu=2):
    netG = Generator(ngpu).to(device)
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
    netG.apply(weights_init)
    netD = Discriminator(ngpu).to(device)
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))
    netD.apply(weights_init)
    print('Initialization successful')
    return netG, netD
```

И обучается по следующему пайплайну:

```
def train_gan(exp_num, criterion=nn.BCELoss(), lr_g=0.0001, lr_d=0.00005, batch_size=64, num_epochs=10):
    netG, netD = initialize()
    criterion = criterion
    fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)
    real_label = 1.
    fake_label = 0.
    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr_d, betas=(beta1, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr_g, betas=(beta1, 0.999))
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    writer = SummaryWriter(log_dir=f'./logs_dcgan/exp_{exp_num}')
    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()
            
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
            
            if i % 1000 == 0:
                writer.add_scalar("iter/Loss_D", errD.item(), iters)
                writer.add_scalar("iter/Loss_G", errG.item(), iters)
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                writer.add_image(f'gen_img_vis_{exp_num}', vutils.make_grid(fake, padding=2, normalize=True), iters)
                
            iters += 1
    
    print("Training is finished!")
    writer.close()
    return fixed_noise, img_list, G_losses, D_losses, netG
```



### 3. Обучение имплементированного GAN

Далее имплементированный ГАН был обучен с лернинг рейт генератора=0.0001, лернинг рейт дискриминатора=0.00005, и бэтчем 64. 

К сожалению, добиться сходимости даже близко не удалось:

![image](https://github.com/victorpakholkov/deep_generative_models_itmo_course/assets/56613496/24028c2c-ec5e-4e67-b80c-93c5f2d7e4e1)

### 4. Эксперименты

После этого, я провел 4 эксперимента с помощью разных техник, но к сожалению, ни в одном случае добиться сходимости не получилось даже близко((.


#### Эксперимент 1.

В ходе первого эксперимента было уменьшено количество сверток в дискриминаторе, а один из слоев был заменен на AvgPool2d. 

Теоретически меньшая по размеру модель должна слабее переобучаться и быть более генерализируемой. 

AvgPool2d слой уменьшает пространственные размеры выходных представлений, сохраняя при этом среднее значение значений пикселей в области пула. Это приводит к уменьшению количества параметров в модели, что делает ее менее сложной и менее склонной к переобучению. Кроме того, слои пулинга, такие как AvgPool2d, могут помочь улучшить стабильность обучения, уменьшая размерность данных и, следовательно, количество вычислений, необходимых для обработки данных.

Теперь дискриминатор выглядит так:

```
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(4, 2, 1),
            nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
```

Однако, по результатам эксперимента на 10 эпохах, с лернинг рейт генератора в 0.0001 и лернинг рейт дискриминатора в 0.00003, размером бача в 64, добиться сходимости так и не удалось.

![image](https://github.com/victorpakholkov/deep_generative_models_itmo_course/assets/56613496/8cc38fb2-2a04-4e39-b5a6-deb0dde0bb42)



#### Эксперимент 2.

Второй эксперимент предполагал замену функции активации в блоке CSPup на LeakyReLU.

Мне показалось это хорошей идеей, потому что LeakyReLU теоретически может улучшить сходимость модели обеспечивая непрерывный градиент вокруг нуля и позволяя нейронам продолжать обучаться даже при отрицательных входных значениях. Это может помочь предотвратить застой в локальных минимумах и улучшить сходимость модели. Кроме того, ReLU имеет проблему "мертвых" нейронов, когда нейроны могут "умереть" и перестать обучаться, если их выход становится равным нулю. Это может привести к тому, что модель не будет обучаться эффективно. LeakyReLU решает эту проблему, позволяя небольшому отрицательному градиенту проходить через нейрон, даже когда он неактивен, что позволяет нейрону продолжать обучаться.

Теперь блок CSPup выглядит вот так:

```
class CSPup(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.kernel_size =kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.split_channels = self.in_channels//2
        self.left = nn.Sequential(
            nn.ConvTranspose2d(self.split_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=False)
        )
        self.right = nn.Sequential(
            nn.Conv2d(self.split_channels, self.out_channels , 1, 1, 0),
            nn.BatchNorm2d(self.out_channels),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(self.out_channels, self.out_channels , self.kernel_size, self.stride, self.padding, bias=False),
            nn.Conv2d(self.out_channels, self.out_channels , 1, 1, 0),
            nn.BatchNorm2d(self.out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.out_channels, self.out_channels, 1, 1, 0),
        )
    def forward(self, x):
        x1 = x[:, :self.split_channels, ...]
        x2 = x[:, self.split_channels:, ...]
        y = self.left(x1)+self.right(x2)
        return y
```

Далее, система была обучена на все тех же 10 эпохах, при все тех же lr_g=0.0001, lr_d=0.0001, batch_size=64.

Однако, добиться сходимости и тут не удалось (мягко говоря):

![image](https://github.com/victorpakholkov/deep_generative_models_itmo_course/assets/56613496/cb4b8c8b-0b7e-4dc8-a0d4-87dbe6251b20)


#### Эксперимент 3.

В третьем эксперименте я решил больше не трогать архитектуру сети, вернув все, как было изначально, а вместо этого поиграться с гиперпараметрами.

Лернинг рейт и генератора, и дискриминатора был оставлен все тот же, а вот размер бэтча был заметно увеличен - до 256. Так же, количество эпох было увеличено с 10 до 15.

Увеличение batch size может помочь улучшить стабильность обучения, поскольку это уменьшает вариации в градиентах между мини-выборками.
Увеличение количества эпох может помочь улучшить точность модели, поскольку это дает модели больше времени для обучения и оптимизации. Это может помочь модели достичь более низкой функции потерь и более высокой точности.

Несмотря на это, и тут результат оказался такой себе:

![image](https://github.com/victorpakholkov/deep_generative_models_itmo_course/assets/56613496/2c3c2836-fc94-400d-8732-ac27aabe831a)


#### Эксперимент 4.

Наконец, в ходе 4го экспиремента, было решено собрать франкенштейна из всех предыдущих варинатов. Бэтч сайз был поставлен на 256, модель тренировалась 15 эпох, вернулись leakyrely в CSPup, было уменьшено количество сверток в дискриминаторе с заменой одного из слоев на AvgPool2d.

Результаты совсем чуть-чуть лучше чем раньше, но все равно произошел конвердженс фейлер и этого самого конвердженс добиться не удалось:

![image](https://github.com/victorpakholkov/deep_generative_models_itmo_course/assets/56613496/ad7878c2-6276-4b36-b93f-b09410314e34)

Результаты чуть-чуть отличаются от шума после прошлых экспериментов:

![image](https://github.com/victorpakholkov/deep_generative_models_itmo_course/assets/56613496/0d76dcf7-c1a7-4de7-85ce-604dad031562)

Добиться сходимости так и не удалось.


#### Размышления после экспериментов

Конечно существует довольно широкий список причин, по которым не удалось достичь сходимости.
Проблема может быть в самой составленной архитектуре модели, и возможно добавление или удаление слоев и изменение их типа помогли бы достичь сходимости модели.

Гиперпараметры, такие как лернинг рейт, размер бэтча , вес регуляризации и т.д., могут иметь большое влияние на обучение модели.
Самый очевидный гиперпараметр тут - количество эпох. 
Если мы не обучаем модель достаточно долго, она может не сходиться. 
Возможно, следовало бы увеличить количество эпох обучения, чтобы дать модели больше времени на сходимость, ибо 10 и 15 эпох, довольно маленькое время, возможно 30-40 эпох было бы более адекватным показателем (но таких вычислительных возможностей у меня, к сожалению, нет).

Наконец, добиться сходимости могли бы помочь изменение функции потерь или тестирование разных способов инициализации весов. 

