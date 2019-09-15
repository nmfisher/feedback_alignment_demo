namespace feedback_alignment

#r "C:\\Users\\nickh\\.nuget\\packages\\mathnet.numerics\\4.8.1\\lib\\netstandard2.0\\MathNet.Numerics.dll";;
#r "C:\\Users\\nickh\\.nuget\\packages\\mathnet.numerics.fsharp\\4.8.1\\lib\\netstandard2.0\\MathNet.Numerics.FSharp.dll";;

open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.Distributions;
open MathNet.Numerics.Random;
open MathNet.Numerics.LinearAlgebra.Factorization

module FeedbackAlignment =
 
  type NN (dim:int) = 
    // weights, orthogonal initialization     let ortho = Matrix<double>.Build.Random(dim * 2, dim + 2).QR().Q;
    let mutable w1 = Matrix<double>.Build.Random(dim, 2); // ortho.SubMatrix(0, dim, 0, 2) // 
    let mutable z1 = Matrix<double>.Build.Random(dim, 1); // first layer activation input
    let mutable w1' = Matrix<double>.Build.Random(dim, 2); // first layer grads
    let mutable w2 = Matrix<double>.Build.Random(dim, dim)// ortho.SubMatrix(dim, dim, 2, dim);//Matrix<double>.Build.Random(dim, dim).QR().Q.Transpose(); // second layer weights
    let mutable w2' = Matrix<double>.Build.Random(dim, dim); // second layer grads 
    let mutable z2 = Matrix<double>.Build.Random(dim, 1); // second layer activation input
    let mutable w3 = Matrix<double>.Build.Random(1, dim); // linear output weights
    let mutable err = 0.0
    let relu x = max x 0.0 
    let relu' x = match x > 0.0 with | true -> 1.0 | false -> 0.0 
    let r2 = Matrix<double>.Build.Random(1, 10)
    let r1 = Matrix<double>.Build.Random(dim, dim)

    member x.Forward (input:Matrix<double>) (output:float) = 
      z1 <- w1 * input |> Matrix.map relu // (dx2) * (2x1) -> (dx1)
      z2 <- w2 * z1 |> Matrix.map relu
      let y_hat = (w3 * z2).Row(0).Item(0)
      if y_hat.Equals(nan) then
        failwith "NAN"
      let loss = (y_hat - output) ** 2.0
      err <- (y_hat - output)
      y_hat
    
    
    member x.Backward (input:Matrix<double>) (symmetric:bool) =
      if symmetric then 
        w2' <- (err * w3.Transpose()).PointwiseMultiply(Matrix.map relu' z2) 
        w1' <- (w2.Transpose() * w2').PointwiseMultiply((Matrix.map relu' z1)) 
      else     
        w2' <- (err * r2.Transpose()).PointwiseMultiply(Matrix.map relu' z2) 
        w1' <- (r1 * w2').PointwiseMultiply((Matrix.map relu' z1)) 

    member x.Update (lr:double) (inputs:Matrix<double>)= 
      w1 <- w1 - (lr * w1' * inputs.Transpose())
      w2 <- w2 - (lr * w2' * z1.Transpose())
      w3 <- w3 - (lr * err * z2.Transpose())

  let nn = NN(10)
  let rnd = System.Random()
  let inputs = [ 0.0,1.0,1.0; 1.0,0.0,1.0; 1.0,1.0,0.0; 0.0,0.0,0.0 ];
  let next () = List.item (rnd.Next inputs.Length) inputs
  let mutable accuracies:float list = []
  let iterations = 5000
  for i in seq { 0..iterations } do 
    let (x1,x2,y1) = next()
    let x = array2D [ [x1; ]; [x2] ]  |> DenseMatrix.ofArray2  
    nn.Forward x y1 |> ignore
    nn.Backward x false
    nn.Update 0.001 x
    if i % 100 = 0 then
      let preds = seq {
        for j in seq { 0..20 } do
        let (x1,x2,y1) = next()
        let pred = nn.Forward (array2D [ [x1;];[x2 ] ] |> DenseMatrix.ofArray2) y1 |> (fun x -> match x > 0.5 with | true -> 1 | _ -> 0)
        if pred = int(y1) then yield true else yield false
      }
      let accuracy = (float(preds |> Seq.where id |> Seq.length) / float(preds |> Seq.length))
      printfn "Accuracy : %f" accuracy
      accuracies <- accuracy :: accuracies
  accuracies |> List.rev    
  