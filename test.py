import pandas as pd
import radar

df = pd.DataFrame({"col1": 1, "col2": 7, "col3": 5, "col4": 10, "col5": 4}, index=[0])

radar.plot(df, facets=df.columns, title="Test")
