use burn::tensor::{Device, Tensor};

//A. Operações elemento a elemento

//B. Exercício: crie duas matrizes de notas e calcule:
//1. A soma das duas matrizes.
//2. A diferença entre elas.
//3. A multiplicação elemento a elemento.
//4. O dobro de cada nota.

pub fn ch_03_basics() {
    let device = Device::flex();

    let notas1 = Tensor::<2>::from_data([[8.0, 7.0], [9.0, 6.0], [5.0, 10.0]], &device);
    let notas2: Tensor<2> = Tensor::<2>::from([[6.0, 2.0], [8.5, 7.5], [6.0, 4.0]]);

    //1. A soma das duas matrizes
    let soma_notas = notas1.clone() + notas2.clone();
    println!("Soma das notas dos tensores = {}", soma_notas);

    //2. A diferença entre elas
    let sub_notas = notas1.clone() - notas2.clone();
    println!("Subtração das notas dos tensores = {}", sub_notas);

    //3. A multiplicaçao de cada elemento
    let multp_notas = notas1.clone() * notas2.clone();
    println!("Multiplicação dos elementos dos tensores = {}", multp_notas);

    //4. O dobro das notas da matriz 1 (notas1)
    let dobro_notas1 = 2.0 * notas1.clone();
    println!("O dobro de notas1 = {}", dobro_notas1);

    //5. A divisão de cada elemento de notas1 por notas2
    let divisao_notas = notas1.clone()/notas2.clone();
    println!("A divisão das matrizes = {}", divisao_notas);
}
