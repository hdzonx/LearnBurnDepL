use burn::tensor::{Device, Tensor};

pub fn ch_02_basiccs() {
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
    println!("Menor nota = {}",menor_nota);

    //4. Calcular a soma das notas
    let soma_notas = notas.clone().sum().to_data();
    println!("Soma das notas = {}", soma_notas);
}
