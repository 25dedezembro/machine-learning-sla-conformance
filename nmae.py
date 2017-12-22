def nmae(test, pred):
  Yi = pd.DataFrame(test)
  MYi = pd.DataFrame(pred)
   
  E = 0.0
  for m in range(0, len(Yi) - 1):
    E += abs((Yi.iloc[m]['DispFrames'] - MYi.iloc[m][0]))
    m += 1
  nmae_resultado = (E/m)/Yi.mean()
  return nmae_resultado
