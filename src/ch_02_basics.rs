use burn::tensor::{Device, Tensor};

pub fn ch_02_basics() {
    //Taredas:
    //1. Calcular a média de todas as notas.
    //2. Encontrar a maior nota.
    //3. Encontrar a menor nota.
    //4. Calcular a soma das notas de cada aluno.
    //5. Calcular a média de cada aluno.
    //6. Transpor a matriz.

    let device = Device::flex();

    let notas = Tensor::<2>::from_data([[8.0, 7.0], [9.0, 6.0], [5.0, 10.0]], &device);

    //1. Calcular a média das notas
    let med_notas = notas.clone().mean().to_data();
    println!("Média de todas as notas = {}", med_notas);

    //2. Encontrar a maior nota
    let maior_nota = notas.clone().max().to_data();
    println!("Maior nota = {}", maior_nota);

    // 3. Encontrar a menor nota
    let menor_nota = notas.clone().min().to_data();
    println!("Menor nota = {}", menor_nota);

    //4. Calcular a soma das notas
    let soma_notas = notas.clone().sum().to_data();
    println!("Soma das notas = {}", soma_notas);

    //5. Calcular a média de cada aluno
    //Aluno 01
    let notas_aluno_01 = notas.clone().slice([0..1, 0..2]);
    println!("\nNotas do aluno 01 = {}", notas_aluno_01);
    let media_notas_alu_01 = notas_aluno_01.mean().to_data();
    println!("Média das notas do aluno 01 = {}", media_notas_alu_01);
    //Aluno 02
    let notas_aluno_02 = notas.clone().slice([1..2, 0..2]);
    println!("\nNotas do aluno 02 = {}", notas_aluno_02);
    let media_notas_alu_02 = notas_aluno_02.mean().to_data();
    println!("Média das notas do aluno 02 ={}", media_notas_alu_02);
    //Aluno 03
    let notas_aluno_03 = notas.clone().slice([2..3, 0..2]);
    println!("\nNotas do aluno 02 = {}", notas_aluno_03);
    let media_notas_alu_03 = notas_aluno_03.mean().to_data();
    println!("Média das notas do aluno 03 ={}", media_notas_alu_03);

    //6. Transpor a matriz
    let transposta = notas.clone().transpose();
    println!("\nTransposta da matriz = {}", transposta);
    //Note que transposta não é o mesmo que reshape:
    let notas_reshape = notas.clone().reshape([2, 3]);
    println!("Reshape = {}", notas_reshape);
}
