import pandas as pd
import radar

df = pd.DataFrame({"col1": 1, "col2": 7, "col3": 5, "col4": 10, "col5": 4}, index=[0])
labs = ["Column 1", "Column 2", "Column 3", "Column 4", "Column 5"]

radar.plot(df, facets=df.columns, title="Test", labels=labs)
