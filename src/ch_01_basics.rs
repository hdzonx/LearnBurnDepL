pub fn basics() {
     use burn::tensor::{Device, Tensor};
        // Seleciona a CPU com a chamada do flex()
    let device = Device::flex();
    //Se for usar GPU, substitua a linha acima por:
    //let device = Device::wgpu(Default::default());

    // Criação dos tensores
    let tensor_1 = Tensor::<2>::from_data([[2.0, 3.0], [4.0, 5.0]], &device);

    let tensor_2 = Tensor::<2>::ones_like(&tensor_1);

    // Soma elemento a elemento na CPU
    println!("{}", tensor_1 + tensor_2);

    //Criando um tensor de notas
    let notas = Tensor::<1>::from_data([6.0, 7.5, 8.0, 5.5, 9.5, 4.0], &device);

    //Total de Elemmentos
    let total_notas = notas.shape().num_elements();
    println!("\nNúmero total de notas = {}", total_notas);

    //Média dos valores
    //Obs: em Burn, o clone() de um tensor geralmente é uma operação de clonagem do handle,
    //não necessariamente uma duplicação física dos dados.
    let nota_media = notas.clone().mean().to_data();
    println!("Média das notas = {}", nota_media);

    //Máximo valor no tensor
    let nota_max = notas.clone().max().to_data();
    println!("Nota máxima = {}", nota_max);

    //Mínimo valor do tensior
    let nota_min = notas.clone().min().to_data();
    println!("Nota mínima = {}", nota_min);

    //Soma dos valores
    let soma_notas = notas.sum().to_data();
    println!("Soma das notas = {}", soma_notas);

    //Tensor bidimesional
    let nota_bid = Tensor::<2>::from_data([[8.0, 7.0], [9.0, 6.5], [5.5, 10.0]], &device);

    //Formato do tensor
    println!("\nShape:{:?}", nota_bid.shape());

    //Total de elementos
    println!(
        "N. total de elementos = {}",
        nota_bid.shape().num_elements()
    );

    // Extrai uma parte do tensor com slice
    let nota_mat_alun_2 = nota_bid.clone().slice([1..2, 0..1]);
    println!(
        "Nota de matemática do aluno 2: {}",
        nota_mat_alun_2.to_data()
    );

    //Transforma o tensor 3x2 em 2x3
    let nota_bid_reshape = nota_bid.reshape([2, 3]);
    println!("Tensor remodelado para 2x3:{}", nota_bid_reshape);

}
