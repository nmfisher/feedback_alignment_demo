namespace feedback_alignment

#r "C:\\Users\\nickh\\.nuget\\packages\\mathnet.numerics\\4.8.1\\lib\\netstandard2.0\\MathNet.Numerics.dll";;
#r "C:\\Users\\nickh\\.nuget\\packages\\mathnet.numerics.fsharp\\4.8.1\\lib\\netstandard2.0\\MathNet.Numerics.FSharp.dll";;

open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.Distributions;
open MathNet.Numerics.Random;

module FeedbackAlignment =
 
  type NN (dim:int) = 
    
    let mutable w1 = Matrix<double>.Build.Random(2, dim); // weights
    let mutable a1 = Matrix<double>.Build.Random(dim, 1); // first layer activation output
    let mutable w1' = Matrix<double>.Build.Random(2, dim); // first layer grads
    let mutable w2 = Matrix<double>.Build.Random(dim, dim); // second layer weights
    let mutable w2' = Matrix<double>.Build.Random(dim, dim); // second layer grads 
    let mutable a2 = Matrix<double>.Build.Random(dim, 1); // second layer activation output
    let mutable w3 = Matrix<double>.Build.Random(1, dim); // linear output weights
    let mutable w3' = Matrix<double>.Build.Random(1, dim); // linear output grads
    let relu x = max x 0.0 
    let relu' x = match x > 0.0 with | true -> 1.0 | false -> 0.0 
    let linear' = id
    let chained = relu' >> linear' 
    let logloss y y' = abs(y - y') // log-loss for unary output

    
    let mutable loss = 0.0// scalar loss
    member x.Loss with get() = loss

    
    member x.ToString () = 
      printfn "W1: %s" (w1.ToString())
      printfn "W2: %s" (w2.ToString())
      printfn "W3: %s" (w3.ToString())
    
    member x.Forward (input:Matrix<double>) (output:float) = 
      // printfn "Forward pass for input %s and output %f" (input.ToString()) output
      let o1 = input.Multiply(w1);
      let a1 = Matrix.map relu o1
      // printfn "a1 : %s" (a1.ToString())
      let o2 = a1 * w2
      let a2 = Matrix.map relu o2
      // printfn "a2 : %s" (a2.ToString())
      let o3 = a2 * output 
      let y' = (o3.Row(0).Item(0))
      // printfn "Prediction : %f" y'
      loss <- logloss y'  output
      // printfn "Loss : %f" loss
      y'
    
    member x.Backward (input:Matrix<double>) =
      // printfn "1"
      w3' <- a2.Transpose() * -1.0 // 1 x 3
      // printfn "2"
      w2' <- Matrix.map relu' a2 |> (fun x -> DenseMatrix.ofDiag (x.Column(0))) // 3 x 1 -> 3 x 3 diagonal
      // printfn "3"
      w2' <- w2' * DenseMatrix.ofDiag (w3.Row(0)) // (3 x 3)  *  (3x3) -> 3x3
      // printfn "4"
      w2' <- w2.Transpose() * w2' //  (3 x 3) * (3 x 3) -> 3x3
      // printfn "5"
      w1' <- Matrix.map relu' a1 |> (fun x -> DenseMatrix.ofDiag (x.Column(0))) // 3 x 1 -> 3 x 3 diagonal
      // printfn "6"
      w1' <- w1 * w1' // (2 x 3) * (3 x 3) -> 2 x 3
      // printfn "7"
      // w1' <- w1' * input
      w1' <- w1' * w2'

    member x.Update (lr:double) = 
      w1 <- w1 - (lr * w1')
      w2 <- w2 - (lr * w2')
      w3 <- w3 - (lr * w3')

  let nn = NN(100)
  let rnd = System.Random()
  let inputs = [ 0.0,1.0,1.0; 1.0,0.0,1.0; 1.0,1.0,0.0; 0.0,0.0,0.0 ];
  let next () = List.item (rnd.Next inputs.Length) inputs
  let mutable accuracies = []
  for i in seq { 0..10000 } do 
    let (x1,x2,y1) = next()
    let x = array2D [ [x1; x2 ] ]  |> DenseMatrix.ofArray2  
    nn.Forward x y1 |> ignore
    nn.Backward x
    nn.Update(0.00001)
    if i % 100 = 0 then
      //nn.ToString()
      let preds = seq {
        for j in seq { 0..20 } do
        let (x1,x2,y1) = next()
        let pred = nn.Forward (array2D [ [x1; x2 ] ]  |> DenseMatrix.ofArray2) y1 |> (fun x -> match x > 0.5 with | true -> 1 | _ -> 0)
        if pred = int(y1) then yield true else yield false
      }
      let accuracy = (float(preds |> Seq.where id |> Seq.length) / float(preds |> Seq.length))
      printfn "Accuracy : %f" accuracy
      accuracies <- accuracy :: accuracies
  printfn "Final average accuracy : %f" (accuracies |> Seq.reduce (+) |> (fun x -> x / float(Seq.length accuracies)))
  