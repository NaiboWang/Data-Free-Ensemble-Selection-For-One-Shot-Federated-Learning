Sub generate_all_tables()
Dim SEMR
SEMR = 26 + 26 + 26 + 26 + 15
sheetname = getSheetName()
generate_pivotTable sheetname, SEMR
generate_f1_table sheetname, SEMR
End Sub
Sub generate_pivotTable(sheetname, SEMR)
'
' generate_pivotTable Macro
'

'
    Worksheets(sheetname).Activate
    num_columns = ActiveSheet.UsedRange.Columns.Count
    Application.CutCopyMode = False
    ' Sheets.Add
    ' ActiveWorkbook.PivotCaches.Create(SourceType:=xlDatabase, SourceData:= _
    '     sheetname & "!R1C1:R1048576C" & num_columns, Version:=8).CreatePivotTable _
    '     TableDestination:="Sheet1!R3C1", TableName:="PivotTable1", DefaultVersion _
    '     :=8
    Sheets.Add
    ActiveWorkbook.PivotCaches.Create(SourceType:=xlDatabase, SourceData:= _
        "exp_results_2212_2112_2016_2018!R1C1:R11943C16", Version:=7). _
        CreatePivotTable TableDestination:="Sheet2!R3C1", TableName:="PivotTable1", _
        DefaultVersion:=7
    Sheets("Sheet1").Select
    Cells(3, 1).Select
    With ActiveSheet.PivotTables("PivotTable1")
        .ColumnGrand = True
        .HasAutoFormat = True
        .DisplayErrorString = False
        .DisplayNullString = True
        .EnableDrilldown = True
        .ErrorString = ""
        .MergeLabels = False
        .NullString = ""
        .PageFieldOrder = 2
        .PageFieldWrapCount = 0
        .PreserveFormatting = True
        .RowGrand = True
        .SaveData = True
        .PrintTitles = False
        .RepeatItemsOnEachPrintedPage = True
        .TotalsAnnotation = False
        .CompactRowIndent = 1
        .InGridDropZones = False
        .DisplayFieldCaptions = True
        .DisplayMemberPropertyTooltips = False
        .DisplayContextTooltips = True
        .ShowDrillIndicators = True
        .PrintDrillIndicators = False
        .AllowMultipleFilters = False
        .SortUsingCustomLists = True
        .FieldListSortAscending = False
        .ShowValuesRow = False
        .CalculatedMembersInFilters = False
        .RowAxisLayout xlCompactRow
    End With
    With ActiveSheet.PivotTables("PivotTable1").PivotCache
        .RefreshOnFileOpen = False
        .MissingItemsLimit = xlMissingItemsDefault
    End With
    ActiveSheet.PivotTables("PivotTable1").RepeatAllLabels xlRepeatLabels
    With ActiveSheet.PivotTables("PivotTable1").PivotFields("model")
        .Orientation = xlRowField
        .Position = 1
    End With
    With ActiveSheet.PivotTables("PivotTable1").PivotFields("method")
        .Orientation = xlRowField
        .Position = 2
    End With
    With ActiveSheet.PivotTables("PivotTable1").PivotFields("partition")
        .Orientation = xlRowField
        .Position = 3
    End With
    With ActiveSheet.PivotTables("PivotTable1").PivotFields("split")
        .Orientation = xlRowField
        .Position = 4
    End With
    With ActiveSheet.PivotTables("PivotTable1").PivotFields("party_num")
        .Orientation = xlRowField
        .Position = 5
    End With
    With ActiveSheet.PivotTables("PivotTable1").PivotFields("K")
        .Orientation = xlRowField
        .Position = 6
    End With
    ActiveSheet.PivotTables("PivotTable1").AddDataField ActiveSheet.PivotTables( _
        "PivotTable1").PivotFields("average_accuracy"), "Max of average_accuracy", _
        xlMax
    With ActiveSheet.PivotTables("PivotTable1").PivotFields("method")
        .Orientation = xlColumnField
        .Position = 1
    End With
    With ActiveSheet.PivotTables("PivotTable1").PivotFields("cluster_method")
        .Orientation = xlColumnField
        .Position = 2
    End With





    Range("DP4").Select
    ActiveCell.FormulaR1C1 = "Best Method"
    Range("DQ4").Select
    ActiveCell.FormulaR1C1 = "Best Hierarchical Method"
    Range("DR4").Select
    ActiveCell.FormulaR1C1 = "Best Accuracy"

    Range("DS4").Select
    ActiveCell.FormulaR1C1 = "Best Method"
    Range("DT4").Select
    ActiveCell.FormulaR1C1 = "BestHierarchical Method"
    Range("DU4").Select
    ActiveCell.FormulaR1C1 = "Best Accuracy"

    Range("DV4").Select
    ActiveCell.FormulaR1C1 = "Best Method"
    Range("DW4").Select
    ActiveCell.FormulaR1C1 = "Best Hierarchical Method"
    Range("DX4").Select
    ActiveCell.FormulaR1C1 = "Best Accuracy"

    Range("DY4").Select
    ActiveCell.FormulaR1C1 = "2th Best method"
    Range("DZ4").Select
    ActiveCell.FormulaR1C1 = "2th Hierarchical Method"
    Range("EA4").Select
    ActiveCell.FormulaR1C1 = "2th Accuracy"

    Range("EB4").Select
    ActiveCell.FormulaR1C1 = "3th Best method"
    Range("EC4").Select
    ActiveCell.FormulaR1C1 = "3th Hierarchical Method"
    Range("ED4").Select
    ActiveCell.FormulaR1C1 = "3th Accuracy"

    Range("EE4").Select
    ActiveCell.FormulaR1C1 = "4th Best method"
    Range("EF4").Select
    ActiveCell.FormulaR1C1 = "4th Hierarchical Method"
    Range("EG4").Select
    ActiveCell.FormulaR1C1 = "4th Accuracy"

    Range("EH4").Select
    ActiveCell.FormulaR1C1 = "5th Best method"
    Range("EI4").Select
    ActiveCell.FormulaR1C1 = "5th Hierarchical Method"
    Range("EJ4").Select
    ActiveCell.FormulaR1C1 = "5th Accuracy"

    Range("EK4").Select
    ActiveCell.FormulaR1C1 = "6th Best method"
    Range("EL4").Select
    ActiveCell.FormulaR1C1 = "6th Hierarchical Method"
    Range("EM4").Select
    ActiveCell.FormulaR1C1 = "6th Accuracy"

    Range("EN4").Select
    ActiveCell.FormulaR1C1 = "7th Best method"
    Range("EO4").Select
    ActiveCell.FormulaR1C1 = "7th Hierarchical Method"
    Range("EP4").Select
    ActiveCell.FormulaR1C1 = "7th Accuracy"

    Range("EQ4").Select
    ActiveCell.FormulaR1C1 = "8th Best method"
    Range("ER4").Select
    ActiveCell.FormulaR1C1 = "8th Hierarchical Method"
    Range("ES4").Select
    ActiveCell.FormulaR1C1 = "8th Accuracy"

    Range("ET4").Select
    ActiveCell.FormulaR1C1 = "9th Best method"
    Range("EU4").Select
    ActiveCell.FormulaR1C1 = "9th Hierarchical Method"
    Range("EV4").Select
    ActiveCell.FormulaR1C1 = "9th Accuracy"

    Range("EW4").Select
    ActiveCell.FormulaR1C1 = "10th Best method"
    Range("EX4").Select
    ActiveCell.FormulaR1C1 = "10th Hierarchical Method"
    Range("EY4").Select
    ActiveCell.FormulaR1C1 = "10th Accuracy"

    Range("EZ4").Select
    ActiveCell.FormulaR1C1 = "11th Best method"
    Range("FA4").Select
    ActiveCell.FormulaR1C1 = "11th Hierarchical Method"
    Range("FB4").Select
    ActiveCell.FormulaR1C1 = "11th Accuracy"

    Range("FC4").Select
    ActiveCell.FormulaR1C1 = "12th Best method"
    Range("FD4").Select
    ActiveCell.FormulaR1C1 = "12th Hierarchical Method"
    Range("FE4").Select
    ActiveCell.FormulaR1C1 = "12th Accuracy"

    Range("FF4").Select
    ActiveCell.FormulaR1C1 = "13th Best method"
    Range("FG4").Select
    ActiveCell.FormulaR1C1 = "13th Hierarchical Method"
    Range("FH4").Select
    ActiveCell.FormulaR1C1 = "13th Accuracy"

    Range("FI4").Select
    ActiveCell.FormulaR1C1 = "14th Best method"
    Range("FJ4").Select
    ActiveCell.FormulaR1C1 = "14th Hierarchical Method"
    Range("FK4").Select
    ActiveCell.FormulaR1C1 = "14th Accuracy"

    Range("FL4").Select
    ActiveCell.FormulaR1C1 = "15th Best method"
    Range("FM4").Select
    ActiveCell.FormulaR1C1 = "15th Hierarchical Method"
    Range("FN4").Select
    ActiveCell.FormulaR1C1 = "15th Accuracy"

    Range("FO4").Select
    ActiveCell.FormulaR1C1 = "16th Best method"
    Range("FP4").Select
    ActiveCell.FormulaR1C1 = "16th Hierarchical Method"
    Range("FQ4").Select
    ActiveCell.FormulaR1C1 = "16th Accuracy"

    Range("FR4").Select
    ActiveCell.FormulaR1C1 = "17th Best method"
    Range("FS4").Select
    ActiveCell.FormulaR1C1 = "17th Hierarchical Method"
    Range("FT4").Select
    ActiveCell.FormulaR1C1 = "17th Accuracy"

    Range("FU4").Select
    ActiveCell.FormulaR1C1 = "18th Best method"
    Range("FV4").Select
    ActiveCell.FormulaR1C1 = "18th Hierarchical Method"
    Range("FW4").Select
    ActiveCell.FormulaR1C1 = "18th Accuracy"

    Range("FX4").Select
    ActiveCell.FormulaR1C1 = "19th Best method"
    Range("FY4").Select
    ActiveCell.FormulaR1C1 = "19th Hierarchical Method"
    Range("FZ4").Select
    ActiveCell.FormulaR1C1 = "19th Accuracy"

    Range("GA4").Select
    ActiveCell.FormulaR1C1 = "20th Best method"
    Range("GB4").Select
    ActiveCell.FormulaR1C1 = "20th Hierarchical Method"
    Range("GC4").Select
    ActiveCell.FormulaR1C1 = "20th Accuracy"

    Range("GD4").Select
    ActiveCell.FormulaR1C1 = "21st Best method"
    Range("GE4").Select
    ActiveCell.FormulaR1C1 = "21st Hierarchical Method"
    Range("GF4").Select
    ActiveCell.FormulaR1C1 = "21st Accuracy"

    Range("GG4").Select
    ActiveCell.FormulaR1C1 = "22nd Best method"
    Range("GH4").Select
    ActiveCell.FormulaR1C1 = "22nd Hierarchical Method"
    Range("GI4").Select
    ActiveCell.FormulaR1C1 = "22nd Accuracy"

    Range("GJ4").Select
    ActiveCell.FormulaR1C1 = "23rd Best method"
    Range("GK4").Select
    ActiveCell.FormulaR1C1 = "23rd Hierarchical Method"
    Range("GL4").Select
    ActiveCell.FormulaR1C1 = "23rd Accuracy"

    Range("GM4").Select
    ActiveCell.FormulaR1C1 = "24th Best method"
    Range("GN4").Select
    ActiveCell.FormulaR1C1 = "24th Hierarchical Method"
    Range("GO4").Select
    ActiveCell.FormulaR1C1 = "24th Accuracy"

    Range("GP4").Select
    ActiveCell.FormulaR1C1 = "25th Best method"
    Range("GQ4").Select
    ActiveCell.FormulaR1C1 = "25th Hierarchical Method"
    Range("GR4").Select
    ActiveCell.FormulaR1C1 = "25th Accuracy"

    Range("GS4").Select
    ActiveCell.FormulaR1C1 = "26th Best method"
    Range("GT4").Select
    ActiveCell.FormulaR1C1 = "26th Hierarchical Method"
    Range("GU4").Select
    ActiveCell.FormulaR1C1 = "26th Accuracy"

    Range("GV4").Select
    ActiveCell.FormulaR1C1 = "27th Best method"
    Range("GW4").Select
    ActiveCell.FormulaR1C1 = "27th Hierarchical Method"
    Range("GX4").Select
    ActiveCell.FormulaR1C1 = "27th Accuracy"

    Range("GY4").Select
    ActiveCell.FormulaR1C1 = "28th Best method"
    Range("GZ4").Select
    ActiveCell.FormulaR1C1 = "28th Hierarchical Method"
    Range("HA4").Select
    ActiveCell.FormulaR1C1 = "28th Accuracy"

    Range("HB4").Select
    ActiveCell.FormulaR1C1 = "29th Best method"
    Range("HC4").Select
    ActiveCell.FormulaR1C1 = "29th Hierarchical Method"
    Range("HD4").Select
    ActiveCell.FormulaR1C1 = "29th Accuracy"

    Range("HE4").Select
    ActiveCell.FormulaR1C1 = "30th Best method"
    Range("HF4").Select
    ActiveCell.FormulaR1C1 = "30th Hierarchical Method"
    Range("HG4").Select
    ActiveCell.FormulaR1C1 = "30th Accuracy"

    Range("HH4").Select
    ActiveCell.FormulaR1C1 = "31th Best method"
    Range("HI4").Select
    ActiveCell.FormulaR1C1 = "31th Hierarchical Method"
    Range("HJ4").Select
    ActiveCell.FormulaR1C1 = "31th Accuracy"
    Range("HK4").Select
    ActiveCell.FormulaR1C1 = "32th Best method"
    Range("HL4").Select
    ActiveCell.FormulaR1C1 = "32th Hierarchical Method"
    Range("HM4").Select
    ActiveCell.FormulaR1C1 = "32th Accuracy"
    Range("HN4").Select
    ActiveCell.FormulaR1C1 = "33th Best method"
    Range("HO4").Select
    ActiveCell.FormulaR1C1 = "33th Hierarchical Method"
    Range("HP4").Select
    ActiveCell.FormulaR1C1 = "33th Accuracy"
    Range("HQ4").Select
    ActiveCell.FormulaR1C1 = "34th Best method"
    Range("HR4").Select
    ActiveCell.FormulaR1C1 = "34th Hierarchical Method"
    Range("HS4").Select
    ActiveCell.FormulaR1C1 = "34th Accuracy"
    Range("HT4").Select
    ActiveCell.FormulaR1C1 = "35th Best method"
    Range("HU4").Select
    ActiveCell.FormulaR1C1 = "35th Hierarchical Method"
    Range("HV4").Select
    ActiveCell.FormulaR1C1 = "35th Accuracy"
    Range("HW4").Select
    ActiveCell.FormulaR1C1 = "36th Best method"
    Range("HX4").Select
    ActiveCell.FormulaR1C1 = "36th Hierarchical Method"
    Range("HY4").Select
    ActiveCell.FormulaR1C1 = "36th Accuracy"
    Range("HZ4").Select
    ActiveCell.FormulaR1C1 = "37th Best method"
    Range("IA4").Select
    ActiveCell.FormulaR1C1 = "37th Hierarchical Method"
    Range("IB4").Select
    ActiveCell.FormulaR1C1 = "37th Accuracy"
    Range("IC4").Select
    ActiveCell.FormulaR1C1 = "38th Best method"
    Range("ID4").Select
    ActiveCell.FormulaR1C1 = "38th Hierarchical Method"
    Range("IE4").Select
    ActiveCell.FormulaR1C1 = "38th Accuracy"
    Range("IF4").Select
    ActiveCell.FormulaR1C1 = "39th Best method"
    Range("IG4").Select
    ActiveCell.FormulaR1C1 = "39th Hierarchical Method"
    Range("IH4").Select
    ActiveCell.FormulaR1C1 = "39th Accuracy"
    Range("II4").Select
    ActiveCell.FormulaR1C1 = "40th Best method"
    Range("IJ4").Select
    ActiveCell.FormulaR1C1 = "40th Hierarchical Method"
    Range("IK4").Select
    ActiveCell.FormulaR1C1 = "40th Accuracy"
    Range("IL4").Select
    ActiveCell.FormulaR1C1 = "41th Best method"
    Range("IM4").Select
    ActiveCell.FormulaR1C1 = "41th Hierarchical Method"
    Range("IN4").Select
    ActiveCell.FormulaR1C1 = "41th Accuracy"
    Range("IO4").Select
    ActiveCell.FormulaR1C1 = "42th Best method"
    Range("IP4").Select
    ActiveCell.FormulaR1C1 = "42th Hierarchical Method"
    Range("IQ4").Select
    ActiveCell.FormulaR1C1 = "42th Accuracy"
    Range("IR4").Select
    ActiveCell.FormulaR1C1 = "43th Best method"
    Range("IS4").Select
    ActiveCell.FormulaR1C1 = "43th Hierarchical Method"
    Range("IT4").Select
    ActiveCell.FormulaR1C1 = "43th Accuracy"
    Range("IU4").Select
    ActiveCell.FormulaR1C1 = "44th Best method"
    Range("IV4").Select
    ActiveCell.FormulaR1C1 = "44th Hierarchical Method"
    Range("IW4").Select
    ActiveCell.FormulaR1C1 = "44th Accuracy"
    Range("IX4").Select
    ActiveCell.FormulaR1C1 = "45th Best method"
    Range("IY4").Select
    ActiveCell.FormulaR1C1 = "45th Hierarchical Method"
    Range("IZ4").Select
    ActiveCell.FormulaR1C1 = "45th Accuracy"
    Range("JA4").Select
    ActiveCell.FormulaR1C1 = "46th Best method"
    Range("JB4").Select
    ActiveCell.FormulaR1C1 = "46th Hierarchical Method"
    Range("JC4").Select
    ActiveCell.FormulaR1C1 = "46th Accuracy"
    Range("JD4").Select
    ActiveCell.FormulaR1C1 = "47th Best method"
    Range("JE4").Select
    ActiveCell.FormulaR1C1 = "47th Hierarchical Method"
    Range("JF4").Select
    ActiveCell.FormulaR1C1 = "47th Accuracy"
    Range("JG4").Select
    ActiveCell.FormulaR1C1 = "48th Best method"
    Range("JH4").Select
    ActiveCell.FormulaR1C1 = "48th Hierarchical Method"
    Range("JI4").Select
    ActiveCell.FormulaR1C1 = "48th Accuracy"
    Range("JJ4").Select
    ActiveCell.FormulaR1C1 = "49th Best method"
    Range("JK4").Select
    ActiveCell.FormulaR1C1 = "49th Hierarchical Method"
    Range("JL4").Select
    ActiveCell.FormulaR1C1 = "49th Accuracy"
    Range("JM4").Select
    ActiveCell.FormulaR1C1 = "50th Best method"
    Range("JN4").Select
    ActiveCell.FormulaR1C1 = "50th Hierarchical Method"
    Range("JO4").Select
    ActiveCell.FormulaR1C1 = "50th Accuracy"
    Range("JP4").Select
    ActiveCell.FormulaR1C1 = "51th Best method"
    Range("JQ4").Select
    ActiveCell.FormulaR1C1 = "51th Hierarchical Method"
    Range("JR4").Select
    ActiveCell.FormulaR1C1 = "51th Accuracy"
    Range("JS4").Select
    ActiveCell.FormulaR1C1 = "52th Best method"
    Range("JT4").Select
    ActiveCell.FormulaR1C1 = "52th Hierarchical Method"
    Range("JU4").Select
    ActiveCell.FormulaR1C1 = "52th Accuracy"
    Range("JV4").Select
    ActiveCell.FormulaR1C1 = "53th Best method"
    Range("JW4").Select
    ActiveCell.FormulaR1C1 = "53th Hierarchical Method"
    Range("JX4").Select
    ActiveCell.FormulaR1C1 = "53th Accuracy"
    Range("JY4").Select
    ActiveCell.FormulaR1C1 = "54th Best method"
    Range("JZ4").Select
    ActiveCell.FormulaR1C1 = "54th Hierarchical Method"
    Range("KA4").Select
    ActiveCell.FormulaR1C1 = "54th Accuracy"
    Range("KB4").Select
    ActiveCell.FormulaR1C1 = "55th Best method"
    Range("KC4").Select
    ActiveCell.FormulaR1C1 = "55th Hierarchical Method"
    Range("KD4").Select
    ActiveCell.FormulaR1C1 = "55th Accuracy"
    Range("KE4").Select
    ActiveCell.FormulaR1C1 = "56th Best method"
    Range("KF4").Select
    ActiveCell.FormulaR1C1 = "56th Hierarchical Method"
    Range("KG4").Select
    ActiveCell.FormulaR1C1 = "56th Accuracy"
    Range("KH4").Select
    ActiveCell.FormulaR1C1 = "57th Best method"
    Range("KI4").Select
    ActiveCell.FormulaR1C1 = "57th Hierarchical Method"
    Range("KJ4").Select
    ActiveCell.FormulaR1C1 = "57th Accuracy"
    Range("KK4").Select
    ActiveCell.FormulaR1C1 = "58th Best method"
    Range("KL4").Select
    ActiveCell.FormulaR1C1 = "58th Hierarchical Method"
    Range("KM4").Select
    ActiveCell.FormulaR1C1 = "58th Accuracy"
    Range("KN4").Select
    ActiveCell.FormulaR1C1 = "59th Best method"
    Range("KO4").Select
    ActiveCell.FormulaR1C1 = "59th Hierarchical Method"
    Range("KP4").Select
    ActiveCell.FormulaR1C1 = "59th Accuracy"
    Range("KQ4").Select
    ActiveCell.FormulaR1C1 = "60th Best method"
    Range("KR4").Select
    ActiveCell.FormulaR1C1 = "60th Hierarchical Method"
    Range("KS4").Select
    ActiveCell.FormulaR1C1 = "60th Accuracy"
    Range("KT4").Select
    ActiveCell.FormulaR1C1 = "61th Best method"
    Range("KU4").Select
    ActiveCell.FormulaR1C1 = "61th Hierarchical Method"
    Range("KV4").Select
    ActiveCell.FormulaR1C1 = "61th Accuracy"
    Range("KW4").Select
    ActiveCell.FormulaR1C1 = "62th Best method"
    Range("KX4").Select
    ActiveCell.FormulaR1C1 = "62th Hierarchical Method"
    Range("KY4").Select
    ActiveCell.FormulaR1C1 = "62th Accuracy"
    Range("KZ4").Select
    ActiveCell.FormulaR1C1 = "63th Best method"
    Range("LA4").Select
    ActiveCell.FormulaR1C1 = "63th Hierarchical Method"
    Range("LB4").Select
    ActiveCell.FormulaR1C1 = "63th Accuracy"
    Range("LC4").Select
    ActiveCell.FormulaR1C1 = "64th Best method"
    Range("LD4").Select
    ActiveCell.FormulaR1C1 = "64th Hierarchical Method"
    Range("LE4").Select
    ActiveCell.FormulaR1C1 = "64th Accuracy"
    Range("LF4").Select
    ActiveCell.FormulaR1C1 = "65th Best method"
    Range("LG4").Select
    ActiveCell.FormulaR1C1 = "65th Hierarchical Method"
    Range("LH4").Select
    ActiveCell.FormulaR1C1 = "65th Accuracy"
    Range("LI4").Select
    ActiveCell.FormulaR1C1 = "66th Best method"
    Range("LJ4").Select
    ActiveCell.FormulaR1C1 = "66th Hierarchical Method"
    Range("LK4").Select
    ActiveCell.FormulaR1C1 = "66th Accuracy"
    Range("LL4").Select
    ActiveCell.FormulaR1C1 = "67th Best method"
    Range("LM4").Select
    ActiveCell.FormulaR1C1 = "67th Hierarchical Method"
    Range("LN4").Select
    ActiveCell.FormulaR1C1 = "67th Accuracy"
    Range("LO4").Select
    ActiveCell.FormulaR1C1 = "68th Best method"
    Range("LP4").Select
    ActiveCell.FormulaR1C1 = "68th Hierarchical Method"
    Range("LQ4").Select
    ActiveCell.FormulaR1C1 = "68th Accuracy"
    Range("LR4").Select
    ActiveCell.FormulaR1C1 = "69th Best method"
    Range("LS4").Select
    ActiveCell.FormulaR1C1 = "69th Hierarchical Method"
    Range("LT4").Select
    ActiveCell.FormulaR1C1 = "69th Accuracy"
    Range("LU4").Select
    ActiveCell.FormulaR1C1 = "70th Best method"
    Range("LV4").Select
    ActiveCell.FormulaR1C1 = "70th Hierarchical Method"
    Range("LW4").Select
    ActiveCell.FormulaR1C1 = "70th Accuracy"
    Range("LX4").Select
    ActiveCell.FormulaR1C1 = "71th Best method"
    Range("LY4").Select
    ActiveCell.FormulaR1C1 = "71th Hierarchical Method"
    Range("LZ4").Select
    ActiveCell.FormulaR1C1 = "71th Accuracy"
    Range("MA4").Select
    ActiveCell.FormulaR1C1 = "72th Best method"
    Range("MB4").Select
    ActiveCell.FormulaR1C1 = "72th Hierarchical Method"
    Range("MC4").Select
    ActiveCell.FormulaR1C1 = "72th Accuracy"
    Range("MD4").Select
    ActiveCell.FormulaR1C1 = "73th Best method"
    Range("ME4").Select
    ActiveCell.FormulaR1C1 = "73th Hierarchical Method"
    Range("MF4").Select
    ActiveCell.FormulaR1C1 = "73th Accuracy"
    Range("MG4").Select
    ActiveCell.FormulaR1C1 = "74th Best method"
    Range("MH4").Select
    ActiveCell.FormulaR1C1 = "74th Hierarchical Method"
    Range("MI4").Select
    ActiveCell.FormulaR1C1 = "74th Accuracy"
    Range("MJ4").Select
    ActiveCell.FormulaR1C1 = "75th Best method"
    Range("MK4").Select
    ActiveCell.FormulaR1C1 = "75th Hierarchical Method"
    Range("ML4").Select
    ActiveCell.FormulaR1C1 = "75th Accuracy"
    Range("MM4").Select
    ActiveCell.FormulaR1C1 = "76th Best method"
    Range("MN4").Select
    ActiveCell.FormulaR1C1 = "76th Hierarchical Method"
    Range("MO4").Select
    ActiveCell.FormulaR1C1 = "76th Accuracy"
    Range("MP4").Select
    ActiveCell.FormulaR1C1 = "77th Best method"
    Range("MQ4").Select
    ActiveCell.FormulaR1C1 = "77th Hierarchical Method"
    Range("MR4").Select
    ActiveCell.FormulaR1C1 = "77th Accuracy"
    Range("MS4").Select
    ActiveCell.FormulaR1C1 = "78th Best method"
    Range("MT4").Select
    ActiveCell.FormulaR1C1 = "78th Hierarchical Method"
    Range("MU4").Select
    ActiveCell.FormulaR1C1 = "78th Accuracy"
    Range("MV4").Select
    ActiveCell.FormulaR1C1 = "79th Best method"
    Range("MW4").Select
    ActiveCell.FormulaR1C1 = "79th Hierarchical Method"
    Range("MX4").Select
    ActiveCell.FormulaR1C1 = "79th Accuracy"
    Range("MY4").Select
    ActiveCell.FormulaR1C1 = "80th Best method"
    Range("MZ4").Select
    ActiveCell.FormulaR1C1 = "80th Hierarchical Method"
    Range("NA4").Select
    ActiveCell.FormulaR1C1 = "80th Accuracy"
    Range("NB4").Select
    ActiveCell.FormulaR1C1 = "81th Best method"
    Range("NC4").Select
    ActiveCell.FormulaR1C1 = "81th Hierarchical Method"
    Range("ND4").Select
    ActiveCell.FormulaR1C1 = "81th Accuracy"
    Range("NE4").Select
    ActiveCell.FormulaR1C1 = "82th Best method"
    Range("NF4").Select
    ActiveCell.FormulaR1C1 = "82th Hierarchical Method"
    Range("NG4").Select
    ActiveCell.FormulaR1C1 = "82th Accuracy"
    Range("NH4").Select
    ActiveCell.FormulaR1C1 = "83th Best method"
    Range("NI4").Select
    ActiveCell.FormulaR1C1 = "83th Hierarchical Method"
    Range("NJ4").Select
    ActiveCell.FormulaR1C1 = "83th Accuracy"
    Range("NK4").Select
    ActiveCell.FormulaR1C1 = "84th Best method"
    Range("NL4").Select
    ActiveCell.FormulaR1C1 = "84th Hierarchical Method"
    Range("NM4").Select
    ActiveCell.FormulaR1C1 = "84th Accuracy"
    Range("NN4").Select
    ActiveCell.FormulaR1C1 = "85th Best method"
    Range("NO4").Select
    ActiveCell.FormulaR1C1 = "85th Hierarchical Method"
    Range("NP4").Select
    ActiveCell.FormulaR1C1 = "85th Accuracy"
    Range("NQ4").Select
    ActiveCell.FormulaR1C1 = "86th Best method"
    Range("NR4").Select
    ActiveCell.FormulaR1C1 = "86th Hierarchical Method"
    Range("NS4").Select
    ActiveCell.FormulaR1C1 = "86th Accuracy"
    Range("NT4").Select
    ActiveCell.FormulaR1C1 = "87th Best method"
    Range("NU4").Select
    ActiveCell.FormulaR1C1 = "87th Hierarchical Method"
    Range("NV4").Select
    ActiveCell.FormulaR1C1 = "87th Accuracy"
    Range("NW4").Select
    ActiveCell.FormulaR1C1 = "88th Best method"
    Range("NX4").Select
    ActiveCell.FormulaR1C1 = "88th Hierarchical Method"
    Range("NY4").Select
    ActiveCell.FormulaR1C1 = "88th Accuracy"
    Range("NZ4").Select
    ActiveCell.FormulaR1C1 = "89th Best method"
    Range("OA4").Select
    ActiveCell.FormulaR1C1 = "89th Hierarchical Method"
    Range("OB4").Select
    ActiveCell.FormulaR1C1 = "89th Accuracy"
    Range("OC4").Select
    ActiveCell.FormulaR1C1 = "90th Best method"
    Range("OD4").Select
    ActiveCell.FormulaR1C1 = "90th Hierarchical Method"
    Range("OE4").Select
    ActiveCell.FormulaR1C1 = "90th Accuracy"
    Range("OF4").Select
    ActiveCell.FormulaR1C1 = "91th Best method"
    Range("OG4").Select
    ActiveCell.FormulaR1C1 = "91th Hierarchical Method"
    Range("OH4").Select
    ActiveCell.FormulaR1C1 = "91th Accuracy"
    Range("OI4").Select
    ActiveCell.FormulaR1C1 = "92th Best method"
    Range("OJ4").Select
    ActiveCell.FormulaR1C1 = "92th Hierarchical Method"
    Range("OK4").Select
    ActiveCell.FormulaR1C1 = "92th Accuracy"
    Range("OL4").Select
    ActiveCell.FormulaR1C1 = "93th Best method"
    Range("OM4").Select
    ActiveCell.FormulaR1C1 = "93th Hierarchical Method"
    Range("ON4").Select
    ActiveCell.FormulaR1C1 = "93th Accuracy"
    Range("OO4").Select
    ActiveCell.FormulaR1C1 = "94th Best method"
    Range("OP4").Select
    ActiveCell.FormulaR1C1 = "94th Hierarchical Method"
    Range("OQ4").Select
    ActiveCell.FormulaR1C1 = "94th Accuracy"
    Range("OR4").Select
    ActiveCell.FormulaR1C1 = "95th Best method"
    Range("OS4").Select
    ActiveCell.FormulaR1C1 = "95th Hierarchical Method"
    Range("OT4").Select
    ActiveCell.FormulaR1C1 = "95th Accuracy"
    Range("OU4").Select
    ActiveCell.FormulaR1C1 = "96th Best method"
    Range("OV4").Select
    ActiveCell.FormulaR1C1 = "96th Hierarchical Method"
    Range("OW4").Select
    ActiveCell.FormulaR1C1 = "96th Accuracy"
    Range("OX4").Select
    ActiveCell.FormulaR1C1 = "97th Best method"
    Range("OY4").Select
    ActiveCell.FormulaR1C1 = "97th Hierarchical Method"
    Range("OZ4").Select
    ActiveCell.FormulaR1C1 = "97th Accuracy"
    Range("PA4").Select
    ActiveCell.FormulaR1C1 = "98th Best method"
    Range("PB4").Select
    ActiveCell.FormulaR1C1 = "98th Hierarchical Method"
    Range("PC4").Select
    ActiveCell.FormulaR1C1 = "98th Accuracy"
    Range("PD4").Select
    ActiveCell.FormulaR1C1 = "99th Best method"
    Range("PE4").Select
    ActiveCell.FormulaR1C1 = "99th Hierarchical Method"
    Range("PF4").Select
    ActiveCell.FormulaR1C1 = "99th Accuracy"
    Range("PG4").Select
    ActiveCell.FormulaR1C1 = "100th Best method"
    Range("PH4").Select
    ActiveCell.FormulaR1C1 = "100th Hierarchical Method"
    Range("PI4").Select
    ActiveCell.FormulaR1C1 = "100th Accuracy"

    Range("DP6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(MAX(RC2:RC" & SEMR & "),RC2:RC" & SEMR & ",0)),"""")"
    Range("DQ6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(MAX(RC2:RC" & SEMR & "),RC2:RC" & SEMR & ",0)),"""")"
    Range("DR6").Select
    ActiveCell.Formula2R1C1 = _
        "=MAX(RC2:RC" & SEMR & ")"

    Range("DS6").Select
    ActiveCell.Formula2R1C1 = _
       "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",2),RC2:RC" & SEMR & ",0)),"""")"
    Range("DT6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",2),RC2:RC" & SEMR & ",0)),"""")"
    Range("DU6").Select
    ActiveCell.Formula2R1C1 = _
        "=LARGE(RC2:RC" & SEMR & ",2)"

    Range("DV6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",3),RC2:RC" & SEMR & ",0)),"""")"
    Range("DW6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",3),RC2:RC" & SEMR & ",0)),"""")"
    Range("DX6").Select
    ActiveCell.Formula2R1C1 = _
        "=LARGE(RC2:RC" & SEMR & ",3)"

    Range("DY6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",4),RC2:RC" & SEMR & ",0)),"""")"
    Range("DZ6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",4),RC2:RC" & SEMR & ",0)),"""")"
    Range("EA6").Select
    ActiveCell.Formula2R1C1 = _
        "=LARGE(RC2:RC" & SEMR & ",4)"

    Range("EB6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",5),RC2:RC" & SEMR & ",0)),"""")"
    Range("EC6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",5),RC2:RC" & SEMR & ",0)),"""")"
    Range("ED6").Select
    ActiveCell.Formula2R1C1 = _
        "=LARGE(RC2:RC" & SEMR & ",5)"

    Range("EE6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",6),RC2:RC" & SEMR & ",0)),"""")"
    Range("EF6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",6),RC2:RC" & SEMR & ",0)),"""")"
    Range("EG6").Select
    ActiveCell.Formula2R1C1 = _
        "=LARGE(RC2:RC" & SEMR & ",6)"

    Range("EH6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",7),RC2:RC" & SEMR & ",0)),"""")"
    Range("EI6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",7),RC2:RC" & SEMR & ",0)),"""")"
    Range("EJ6").Select
    ActiveCell.Formula2R1C1 = _
        "=LARGE(RC2:RC" & SEMR & ",7)"

    Range("EK6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",8),RC2:RC" & SEMR & ",0)),"""")"
    Range("EL6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",8),RC2:RC" & SEMR & ",0)),"""")"
    Range("EM6").Select
    ActiveCell.Formula2R1C1 = _
        "=LARGE(RC2:RC" & SEMR & ",8)"

    Range("EN6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",9),RC2:RC" & SEMR & ",0)),"""")"
    Range("EO6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",9),RC2:RC" & SEMR & ",0)),"""")"
    Range("EP6").Select
    ActiveCell.Formula2R1C1 = _
        "=LARGE(RC2:RC" & SEMR & ",9)"

    Range("EQ6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",10),RC2:RC" & SEMR & ",0)),"""")"
    Range("ER6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",10),RC2:RC" & SEMR & ",0)),"""")"
    Range("ES6").Select
    ActiveCell.Formula2R1C1 = _
        "=LARGE(RC2:RC" & SEMR & ",10)"

    Range("ET6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",11),RC2:RC" & SEMR & ",0)),"""")"
    Range("EU6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",11),RC2:RC" & SEMR & ",0)),"""")"
    Range("EV6").Select
    ActiveCell.Formula2R1C1 = _
        "=LARGE(RC2:RC" & SEMR & ",11)"

    Range("EW6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",12),RC2:RC" & SEMR & ",0)),"""")"
    Range("EX6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",12),RC2:RC" & SEMR & ",0)),"""")"
    Range("EY6").Select
    ActiveCell.Formula2R1C1 = _
        "=LARGE(RC2:RC" & SEMR & ",12)"

    Range("EZ6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",13),RC2:RC" & SEMR & ",0)),"""")"
    Range("FA6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",13),RC2:RC" & SEMR & ",0)),"""")"
    Range("FB6").Select
    ActiveCell.Formula2R1C1 = _
        "=LARGE(RC2:RC" & SEMR & ",13)"

    Range("FC6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",14),RC2:RC" & SEMR & ",0)),"""")"
    Range("FD6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",14),RC2:RC" & SEMR & ",0)),"""")"
    Range("FE6").Select
    ActiveCell.Formula2R1C1 = _
        "=LARGE(RC2:RC" & SEMR & ",14)"

    Range("FF6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",15),RC2:RC" & SEMR & ",0)),"""")"
    Range("FG6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",15),RC2:RC" & SEMR & ",0)),"""")"
    Range("FH6").Select
    ActiveCell.Formula2R1C1 = _
        "=LARGE(RC2:RC" & SEMR & ",15)"

    Range("FI6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",16),RC2:RC" & SEMR & ",0)),"""")"
    Range("FJ6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",16),RC2:RC" & SEMR & ",0)),"""")"
    Range("FK6").Select
    ActiveCell.Formula2R1C1 = _
        "=LARGE(RC2:RC" & SEMR & ",16)"


    Range("FL6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",17),RC2:RC" & SEMR & ",0)),"""")"
    Range("FM6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",17),RC2:RC" & SEMR & ",0)),"""")"
    Range("FN6").Select
    ActiveCell.Formula2R1C1 = _
        "=LARGE(RC2:RC" & SEMR & ",17)"


    Range("FO6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",18),RC2:RC" & SEMR & ",0)),"""")"
    Range("FP6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",18),RC2:RC" & SEMR & ",0)),"""")"
    Range("FQ6").Select
    ActiveCell.Formula2R1C1 = _
        "=LARGE(RC2:RC" & SEMR & ",18)"


    Range("FR6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",19),RC2:RC" & SEMR & ",0)),"""")"
    Range("FS6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",19),RC2:RC" & SEMR & ",0)),"""")"
    Range("FT6").Select
    ActiveCell.Formula2R1C1 = _
        "=LARGE(RC2:RC" & SEMR & ",19)"


    Range("FU6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",20),RC2:RC" & SEMR & ",0)),"""")"
    Range("FV6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",20),RC2:RC" & SEMR & ",0)),"""")"
    Range("FW6").Select
    ActiveCell.Formula2R1C1 = _
        "=LARGE(RC2:RC" & SEMR & ",20)"

    Range("FX6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",21),RC2:RC" & SEMR & ",0)),"""")"
    Range("FY6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",21),RC2:RC" & SEMR & ",0)),"""")"
    Range("FZ6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",21)"
    Range("GA6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",22),RC2:RC" & SEMR & ",0)),"""")"
    Range("GB6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",22),RC2:RC" & SEMR & ",0)),"""")"
    Range("GC6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",22)"
    Range("GD6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",23),RC2:RC" & SEMR & ",0)),"""")"
    Range("GE6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",23),RC2:RC" & SEMR & ",0)),"""")"
    Range("GF6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",23)"
    Range("GG6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",24),RC2:RC" & SEMR & ",0)),"""")"
    Range("GH6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",24),RC2:RC" & SEMR & ",0)),"""")"
    Range("GI6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",24)"
    Range("GJ6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",25),RC2:RC" & SEMR & ",0)),"""")"
    Range("GK6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",25),RC2:RC" & SEMR & ",0)),"""")"
    Range("GL6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",25)"
    Range("GM6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",26),RC2:RC" & SEMR & ",0)),"""")"
    Range("GN6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",26),RC2:RC" & SEMR & ",0)),"""")"
    Range("GO6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",26)"
    Range("GP6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",27),RC2:RC" & SEMR & ",0)),"""")"
    Range("GQ6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",27),RC2:RC" & SEMR & ",0)),"""")"
    Range("GR6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",27)"
    Range("GS6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",28),RC2:RC" & SEMR & ",0)),"""")"
    Range("GT6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",28),RC2:RC" & SEMR & ",0)),"""")"
    Range("GU6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",28)"
    Range("GV6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",29),RC2:RC" & SEMR & ",0)),"""")"
    Range("GW6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",29),RC2:RC" & SEMR & ",0)),"""")"
    Range("GX6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",29)"
    Range("GY6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",30),RC2:RC" & SEMR & ",0)),"""")"
    Range("GZ6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",30),RC2:RC" & SEMR & ",0)),"""")"
    Range("HA6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",30)"
    Range("HB6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",31),RC2:RC" & SEMR & ",0)),"""")"
    Range("HC6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",31),RC2:RC" & SEMR & ",0)),"""")"
    Range("HD6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",31)"
    Range("HE6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",32),RC2:RC" & SEMR & ",0)),"""")"
    Range("HF6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",32),RC2:RC" & SEMR & ",0)),"""")"
    Range("HG6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",32)"
    Range("HH6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",33),RC2:RC" & SEMR & ",0)),"""")"
    Range("HI6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",33),RC2:RC" & SEMR & ",0)),"""")"
    Range("HJ6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",33)"
    Range("HK6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",34),RC2:RC" & SEMR & ",0)),"""")"
    Range("HL6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",34),RC2:RC" & SEMR & ",0)),"""")"
    Range("HM6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",34)"
    Range("HN6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",35),RC2:RC" & SEMR & ",0)),"""")"
    Range("HO6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",35),RC2:RC" & SEMR & ",0)),"""")"
    Range("HP6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",35)"
    Range("HQ6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",36),RC2:RC" & SEMR & ",0)),"""")"
    Range("HR6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",36),RC2:RC" & SEMR & ",0)),"""")"
    Range("HS6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",36)"
    Range("HT6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",37),RC2:RC" & SEMR & ",0)),"""")"
    Range("HU6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",37),RC2:RC" & SEMR & ",0)),"""")"
    Range("HV6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",37)"
    Range("HW6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",38),RC2:RC" & SEMR & ",0)),"""")"
    Range("HX6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",38),RC2:RC" & SEMR & ",0)),"""")"
    Range("HY6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",38)"
    Range("HZ6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",39),RC2:RC" & SEMR & ",0)),"""")"
    Range("IA6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",39),RC2:RC" & SEMR & ",0)),"""")"
    Range("IB6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",39)"
    Range("IC6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",40),RC2:RC" & SEMR & ",0)),"""")"
    Range("ID6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",40),RC2:RC" & SEMR & ",0)),"""")"
    Range("IE6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",40)"
    Range("IF6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",41),RC2:RC" & SEMR & ",0)),"""")"
    Range("IG6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",41),RC2:RC" & SEMR & ",0)),"""")"
    Range("IH6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",41)"
    Range("II6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",42),RC2:RC" & SEMR & ",0)),"""")"
    Range("IJ6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",42),RC2:RC" & SEMR & ",0)),"""")"
    Range("IK6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",42)"
    Range("IL6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",43),RC2:RC" & SEMR & ",0)),"""")"
    Range("IM6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",43),RC2:RC" & SEMR & ",0)),"""")"
    Range("IN6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",43)"
    Range("IO6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",44),RC2:RC" & SEMR & ",0)),"""")"
    Range("IP6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",44),RC2:RC" & SEMR & ",0)),"""")"
    Range("IQ6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",44)"
    Range("IR6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",45),RC2:RC" & SEMR & ",0)),"""")"
    Range("IS6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",45),RC2:RC" & SEMR & ",0)),"""")"
    Range("IT6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",45)"
    Range("IU6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",46),RC2:RC" & SEMR & ",0)),"""")"
    Range("IV6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",46),RC2:RC" & SEMR & ",0)),"""")"
    Range("IW6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",46)"
    Range("IX6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",47),RC2:RC" & SEMR & ",0)),"""")"
    Range("IY6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",47),RC2:RC" & SEMR & ",0)),"""")"
    Range("IZ6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",47)"
    Range("JA6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",48),RC2:RC" & SEMR & ",0)),"""")"
    Range("JB6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",48),RC2:RC" & SEMR & ",0)),"""")"
    Range("JC6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",48)"
    Range("JD6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",49),RC2:RC" & SEMR & ",0)),"""")"
    Range("JE6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",49),RC2:RC" & SEMR & ",0)),"""")"
    Range("JF6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",49)"
    Range("JG6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",50),RC2:RC" & SEMR & ",0)),"""")"
    Range("JH6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",50),RC2:RC" & SEMR & ",0)),"""")"
    Range("JI6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",50)"
    Range("JJ6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",51),RC2:RC" & SEMR & ",0)),"""")"
    Range("JK6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",51),RC2:RC" & SEMR & ",0)),"""")"
    Range("JL6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",51)"
    Range("JM6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",52),RC2:RC" & SEMR & ",0)),"""")"
    Range("JN6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",52),RC2:RC" & SEMR & ",0)),"""")"
    Range("JO6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",52)"
    Range("JP6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",53),RC2:RC" & SEMR & ",0)),"""")"
    Range("JQ6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",53),RC2:RC" & SEMR & ",0)),"""")"
    Range("JR6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",53)"
    Range("JS6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",54),RC2:RC" & SEMR & ",0)),"""")"
    Range("JT6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",54),RC2:RC" & SEMR & ",0)),"""")"
    Range("JU6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",54)"
    Range("JV6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",55),RC2:RC" & SEMR & ",0)),"""")"
    Range("JW6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",55),RC2:RC" & SEMR & ",0)),"""")"
    Range("JX6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",55)"
    Range("JY6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",56),RC2:RC" & SEMR & ",0)),"""")"
    Range("JZ6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",56),RC2:RC" & SEMR & ",0)),"""")"
    Range("KA6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",56)"
    Range("KB6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",57),RC2:RC" & SEMR & ",0)),"""")"
    Range("KC6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",57),RC2:RC" & SEMR & ",0)),"""")"
    Range("KD6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",57)"
    Range("KE6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",58),RC2:RC" & SEMR & ",0)),"""")"
    Range("KF6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",58),RC2:RC" & SEMR & ",0)),"""")"
    Range("KG6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",58)"
    Range("KH6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",59),RC2:RC" & SEMR & ",0)),"""")"
    Range("KI6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",59),RC2:RC" & SEMR & ",0)),"""")"
    Range("KJ6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",59)"
    Range("KK6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",60),RC2:RC" & SEMR & ",0)),"""")"
    Range("KL6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",60),RC2:RC" & SEMR & ",0)),"""")"
    Range("KM6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",60)"
    Range("KN6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",61),RC2:RC" & SEMR & ",0)),"""")"
    Range("KO6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",61),RC2:RC" & SEMR & ",0)),"""")"
    Range("KP6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",61)"
    Range("KQ6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",62),RC2:RC" & SEMR & ",0)),"""")"
    Range("KR6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",62),RC2:RC" & SEMR & ",0)),"""")"
    Range("KS6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",62)"
    Range("KT6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",63),RC2:RC" & SEMR & ",0)),"""")"
    Range("KU6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",63),RC2:RC" & SEMR & ",0)),"""")"
    Range("KV6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",63)"
    Range("KW6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",64),RC2:RC" & SEMR & ",0)),"""")"
    Range("KX6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",64),RC2:RC" & SEMR & ",0)),"""")"
    Range("KY6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",64)"
    Range("KZ6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",65),RC2:RC" & SEMR & ",0)),"""")"
    Range("LA6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",65),RC2:RC" & SEMR & ",0)),"""")"
    Range("LB6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",65)"
    Range("LC6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",66),RC2:RC" & SEMR & ",0)),"""")"
    Range("LD6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",66),RC2:RC" & SEMR & ",0)),"""")"
    Range("LE6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",66)"
    Range("LF6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",67),RC2:RC" & SEMR & ",0)),"""")"
    Range("LG6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",67),RC2:RC" & SEMR & ",0)),"""")"
    Range("LH6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",67)"
    Range("LI6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",68),RC2:RC" & SEMR & ",0)),"""")"
    Range("LJ6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",68),RC2:RC" & SEMR & ",0)),"""")"
    Range("LK6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",68)"
    Range("LL6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",69),RC2:RC" & SEMR & ",0)),"""")"
    Range("LM6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",69),RC2:RC" & SEMR & ",0)),"""")"
    Range("LN6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",69)"
    Range("LO6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",70),RC2:RC" & SEMR & ",0)),"""")"
    Range("LP6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",70),RC2:RC" & SEMR & ",0)),"""")"
    Range("LQ6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",70)"
    Range("LR6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",71),RC2:RC" & SEMR & ",0)),"""")"
    Range("LS6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",71),RC2:RC" & SEMR & ",0)),"""")"
    Range("LT6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",71)"
    Range("LU6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",72),RC2:RC" & SEMR & ",0)),"""")"
    Range("LV6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",72),RC2:RC" & SEMR & ",0)),"""")"
    Range("LW6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",72)"
    Range("LX6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",73),RC2:RC" & SEMR & ",0)),"""")"
    Range("LY6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",73),RC2:RC" & SEMR & ",0)),"""")"
    Range("LZ6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",73)"
    Range("MA6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",74),RC2:RC" & SEMR & ",0)),"""")"
    Range("MB6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",74),RC2:RC" & SEMR & ",0)),"""")"
    Range("MC6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",74)"
    Range("MD6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",75),RC2:RC" & SEMR & ",0)),"""")"
    Range("ME6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",75),RC2:RC" & SEMR & ",0)),"""")"
    Range("MF6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",75)"
    Range("MG6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",76),RC2:RC" & SEMR & ",0)),"""")"
    Range("MH6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",76),RC2:RC" & SEMR & ",0)),"""")"
    Range("MI6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",76)"
    Range("MJ6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",77),RC2:RC" & SEMR & ",0)),"""")"
    Range("MK6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",77),RC2:RC" & SEMR & ",0)),"""")"
    Range("ML6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",77)"
    Range("MM6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",78),RC2:RC" & SEMR & ",0)),"""")"
    Range("MN6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",78),RC2:RC" & SEMR & ",0)),"""")"
    Range("MO6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",78)"
    Range("MP6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",79),RC2:RC" & SEMR & ",0)),"""")"
    Range("MQ6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",79),RC2:RC" & SEMR & ",0)),"""")"
    Range("MR6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",79)"
    Range("MS6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",80),RC2:RC" & SEMR & ",0)),"""")"
    Range("MT6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",80),RC2:RC" & SEMR & ",0)),"""")"
    Range("MU6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",80)"
    Range("MV6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",81),RC2:RC" & SEMR & ",0)),"""")"
    Range("MW6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",81),RC2:RC" & SEMR & ",0)),"""")"
    Range("MX6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",81)"
    Range("MY6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",82),RC2:RC" & SEMR & ",0)),"""")"
    Range("MZ6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",82),RC2:RC" & SEMR & ",0)),"""")"
    Range("NA6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",82)"
    Range("NB6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",83),RC2:RC" & SEMR & ",0)),"""")"
    Range("NC6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",83),RC2:RC" & SEMR & ",0)),"""")"
    Range("ND6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",83)"
    Range("NE6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",84),RC2:RC" & SEMR & ",0)),"""")"
    Range("NF6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",84),RC2:RC" & SEMR & ",0)),"""")"
    Range("NG6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",84)"
    Range("NH6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",85),RC2:RC" & SEMR & ",0)),"""")"
    Range("NI6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",85),RC2:RC" & SEMR & ",0)),"""")"
    Range("NJ6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",85)"
    Range("NK6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",86),RC2:RC" & SEMR & ",0)),"""")"
    Range("NL6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",86),RC2:RC" & SEMR & ",0)),"""")"
    Range("NM6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",86)"
    Range("NN6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",87),RC2:RC" & SEMR & ",0)),"""")"
    Range("NO6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",87),RC2:RC" & SEMR & ",0)),"""")"
    Range("NP6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",87)"
    Range("NQ6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",88),RC2:RC" & SEMR & ",0)),"""")"
    Range("NR6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",88),RC2:RC" & SEMR & ",0)),"""")"
    Range("NS6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",88)"
    Range("NT6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",89),RC2:RC" & SEMR & ",0)),"""")"
    Range("NU6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",89),RC2:RC" & SEMR & ",0)),"""")"
    Range("NV6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",89)"
    Range("NW6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",90),RC2:RC" & SEMR & ",0)),"""")"
    Range("NX6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",90),RC2:RC" & SEMR & ",0)),"""")"
    Range("NY6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",90)"
    Range("NZ6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",91),RC2:RC" & SEMR & ",0)),"""")"
    Range("OA6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",91),RC2:RC" & SEMR & ",0)),"""")"
    Range("OB6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",91)"
    Range("OC6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",92),RC2:RC" & SEMR & ",0)),"""")"
    Range("OD6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",92),RC2:RC" & SEMR & ",0)),"""")"
    Range("OE6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",92)"
    Range("OF6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",93),RC2:RC" & SEMR & ",0)),"""")"
    Range("OG6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",93),RC2:RC" & SEMR & ",0)),"""")"
    Range("OH6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",93)"
    Range("OI6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",94),RC2:RC" & SEMR & ",0)),"""")"
    Range("OJ6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",94),RC2:RC" & SEMR & ",0)),"""")"
    Range("OK6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",94)"
    Range("OL6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",95),RC2:RC" & SEMR & ",0)),"""")"
    Range("OM6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",95),RC2:RC" & SEMR & ",0)),"""")"
    Range("ON6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",95)"
    Range("OO6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",96),RC2:RC" & SEMR & ",0)),"""")"
    Range("OP6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",96),RC2:RC" & SEMR & ",0)),"""")"
    Range("OQ6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",96)"
    Range("OR6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",97),RC2:RC" & SEMR & ",0)),"""")"
    Range("OS6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",97),RC2:RC" & SEMR & ",0)),"""")"
    Range("OT6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",97)"
    Range("OU6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",98),RC2:RC" & SEMR & ",0)),"""")"
    Range("OV6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",98),RC2:RC" & SEMR & ",0)),"""")"
    Range("OW6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",98)"
    Range("OX6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",99),RC2:RC" & SEMR & ",0)),"""")"
    Range("OY6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",99),RC2:RC" & SEMR & ",0)),"""")"
    Range("OZ6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",99)"
    Range("PA6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",100),RC2:RC" & SEMR & ",0)),"""")"
    Range("PB6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",100),RC2:RC" & SEMR & ",0)),"""")"
    Range("PC6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",100)"
    Range("PD6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",101),RC2:RC" & SEMR & ",0)),"""")"
    Range("PE6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",101),RC2:RC" & SEMR & ",0)),"""")"
    Range("PF6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",101)"
    Range("PG6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",102),RC2:RC" & SEMR & ",0)),"""")"
    Range("PH6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",102),RC2:RC" & SEMR & ",0)),"""")"
    Range("PI6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",102)"

    Range("DP6:PI6").Select
    Selection.AutoFill Destination:=Range("DP6:PI880")

    Columns("DP:PI").Select
    With Selection
        .HorizontalAlignment = xlCenter
        .VerticalAlignment = xlBottom
        .WrapText = False
        .Orientation = 0
        .AddIndent = False
        .IndentLevel = 0
        .ShrinkToFit = False
        .ReadingOrder = xlContext
        .MergeCells = False
    End With
    Columns("DP:PI").EntireColumn.AutoFit
    Columns("B:AO").Select
    Range("AO1").Activate
    Selection.ColumnWidth = 11.4
    Range("AO3").Select
    Selection.Copy
    Range("DP5:PI5").Select
    Selection.PasteSpecial Paste:=xlPasteFormats, Operation:=xlNone, _
        SkipBlanks:=False, Transpose:=False
    Application.CutCopyMode = False
    Range("B6").Select
    ActiveWindow.FreezePanes = True
    ActiveWindow.SmallScroll Down:=0
    Columns("A:PI").Select
    sF = "=ISNUMBER($A1)"
    For Each oFC In Selection.FormatConditions
        oFC.Delete
    Next
    Set oFC = Selection.FormatConditions.Add(xlExpression, , sF)
    With oFC
        '设置条件格式的填充色
        .Interior.Color = RGB(221, 235, 247)
    End With

    Sheets("Sheet1").Select
    Sheets("Sheet1").Name = "Accuracy"

End Sub

Sub generate_f1_table(sheetname, SEMR)
'F1-Score Table
    Worksheets(sheetname).Activate
    num_columns = ActiveSheet.UsedRange.Columns.Count

    Sheets.Add
    ActiveWorkbook.PivotCaches.Create(SourceType:=xlDatabase, SourceData:= _
        sheetname & "!R1C1:R1048576C" & num_columns, Version:=8).CreatePivotTable _
        TableDestination:="Sheet2!R3C1", TableName:="PivotTable2", DefaultVersion _
        :=8
    Sheets("Sheet2").Select
    Cells(3, 1).Select
    With ActiveSheet.PivotTables("PivotTable2")
        .ColumnGrand = True
        .HasAutoFormat = True
        .DisplayErrorString = False
        .DisplayNullString = True
        .EnableDrilldown = True
        .ErrorString = ""
        .MergeLabels = False
        .NullString = ""
        .PageFieldOrder = 2
        .PageFieldWrapCount = 0
        .PreserveFormatting = True
        .RowGrand = True
        .SaveData = True
        .PrintTitles = False
        .RepeatItemsOnEachPrintedPage = True
        .TotalsAnnotation = False
        .CompactRowIndent = 1
        .InGridDropZones = False
        .DisplayFieldCaptions = True
        .DisplayMemberPropertyTooltips = False
        .DisplayContextTooltips = True
        .ShowDrillIndicators = True
        .PrintDrillIndicators = False
        .AllowMultipleFilters = False
        .SortUsingCustomLists = True
        .FieldListSortAscending = False
        .ShowValuesRow = False
        .CalculatedMembersInFilters = False
        .RowAxisLayout xlCompactRow
    End With
    With ActiveSheet.PivotTables("PivotTable2").PivotCache
        .RefreshOnFileOpen = False
        .MissingItemsLimit = xlMissingItemsDefault
    End With
    ActiveSheet.PivotTables("PivotTable2").RepeatAllLabels xlRepeatLabels
    With ActiveSheet.PivotTables("PivotTable2").PivotFields("model")
        .Orientation = xlRowField
        .Position = 1
    End With
    With ActiveSheet.PivotTables("PivotTable2").PivotFields("method")
        .Orientation = xlRowField
        .Position = 2
    End With
    With ActiveSheet.PivotTables("PivotTable2").PivotFields("partition")
        .Orientation = xlRowField
        .Position = 3
    End With
    With ActiveSheet.PivotTables("PivotTable2").PivotFields("split")
        .Orientation = xlRowField
        .Position = 4
    End With
    With ActiveSheet.PivotTables("PivotTable2").PivotFields("party_num")
        .Orientation = xlRowField
        .Position = 5
    End With
    With ActiveSheet.PivotTables("PivotTable2").PivotFields("K")
        .Orientation = xlRowField
        .Position = 6
    End With
    ActiveSheet.PivotTables("PivotTable2").AddDataField ActiveSheet.PivotTables( _
        "PivotTable2").PivotFields("average_fscore"), "Max of average_fscore", _
        xlMax
    With ActiveSheet.PivotTables("PivotTable2").PivotFields("method")
        .Orientation = xlColumnField
        .Position = 1
    End With
    With ActiveSheet.PivotTables("PivotTable2").PivotFields("cluster_method")
        .Orientation = xlColumnField
        .Position = 2
    End With

 Range("DP4").Select
    ActiveCell.FormulaR1C1 = "Best Method"
    Range("DQ4").Select
    ActiveCell.FormulaR1C1 = "Best Hierarchical Method"
    Range("DR4").Select
    ActiveCell.FormulaR1C1 = "Best Accuracy"

    Range("DS4").Select
    ActiveCell.FormulaR1C1 = "Best Method"
    Range("DT4").Select
    ActiveCell.FormulaR1C1 = "BestHierarchical Method"
    Range("DU4").Select
    ActiveCell.FormulaR1C1 = "Best Accuracy"

    Range("DV4").Select
    ActiveCell.FormulaR1C1 = "Best Method"
    Range("DW4").Select
    ActiveCell.FormulaR1C1 = "Best Hierarchical Method"
    Range("DX4").Select
    ActiveCell.FormulaR1C1 = "Best Accuracy"

    Range("DY4").Select
    ActiveCell.FormulaR1C1 = "2th Best method"
    Range("DZ4").Select
    ActiveCell.FormulaR1C1 = "2th Hierarchical Method"
    Range("EA4").Select
    ActiveCell.FormulaR1C1 = "2th Accuracy"

    Range("EB4").Select
    ActiveCell.FormulaR1C1 = "3th Best method"
    Range("EC4").Select
    ActiveCell.FormulaR1C1 = "3th Hierarchical Method"
    Range("ED4").Select
    ActiveCell.FormulaR1C1 = "3th Accuracy"

    Range("EE4").Select
    ActiveCell.FormulaR1C1 = "4th Best method"
    Range("EF4").Select
    ActiveCell.FormulaR1C1 = "4th Hierarchical Method"
    Range("EG4").Select
    ActiveCell.FormulaR1C1 = "4th Accuracy"

    Range("EH4").Select
    ActiveCell.FormulaR1C1 = "5th Best method"
    Range("EI4").Select
    ActiveCell.FormulaR1C1 = "5th Hierarchical Method"
    Range("EJ4").Select
    ActiveCell.FormulaR1C1 = "5th Accuracy"

    Range("EK4").Select
    ActiveCell.FormulaR1C1 = "6th Best method"
    Range("EL4").Select
    ActiveCell.FormulaR1C1 = "6th Hierarchical Method"
    Range("EM4").Select
    ActiveCell.FormulaR1C1 = "6th Accuracy"

    Range("EN4").Select
    ActiveCell.FormulaR1C1 = "7th Best method"
    Range("EO4").Select
    ActiveCell.FormulaR1C1 = "7th Hierarchical Method"
    Range("EP4").Select
    ActiveCell.FormulaR1C1 = "7th Accuracy"

    Range("EQ4").Select
    ActiveCell.FormulaR1C1 = "8th Best method"
    Range("ER4").Select
    ActiveCell.FormulaR1C1 = "8th Hierarchical Method"
    Range("ES4").Select
    ActiveCell.FormulaR1C1 = "8th Accuracy"

    Range("ET4").Select
    ActiveCell.FormulaR1C1 = "9th Best method"
    Range("EU4").Select
    ActiveCell.FormulaR1C1 = "9th Hierarchical Method"
    Range("EV4").Select
    ActiveCell.FormulaR1C1 = "9th Accuracy"

    Range("EW4").Select
    ActiveCell.FormulaR1C1 = "10th Best method"
    Range("EX4").Select
    ActiveCell.FormulaR1C1 = "10th Hierarchical Method"
    Range("EY4").Select
    ActiveCell.FormulaR1C1 = "10th Accuracy"

    Range("EZ4").Select
    ActiveCell.FormulaR1C1 = "11th Best method"
    Range("FA4").Select
    ActiveCell.FormulaR1C1 = "11th Hierarchical Method"
    Range("FB4").Select
    ActiveCell.FormulaR1C1 = "11th Accuracy"

    Range("FC4").Select
    ActiveCell.FormulaR1C1 = "12th Best method"
    Range("FD4").Select
    ActiveCell.FormulaR1C1 = "12th Hierarchical Method"
    Range("FE4").Select
    ActiveCell.FormulaR1C1 = "12th Accuracy"

    Range("FF4").Select
    ActiveCell.FormulaR1C1 = "13th Best method"
    Range("FG4").Select
    ActiveCell.FormulaR1C1 = "13th Hierarchical Method"
    Range("FH4").Select
    ActiveCell.FormulaR1C1 = "13th Accuracy"

    Range("FI4").Select
    ActiveCell.FormulaR1C1 = "14th Best method"
    Range("FJ4").Select
    ActiveCell.FormulaR1C1 = "14th Hierarchical Method"
    Range("FK4").Select
    ActiveCell.FormulaR1C1 = "14th Accuracy"

    Range("FL4").Select
    ActiveCell.FormulaR1C1 = "15th Best method"
    Range("FM4").Select
    ActiveCell.FormulaR1C1 = "15th Hierarchical Method"
    Range("FN4").Select
    ActiveCell.FormulaR1C1 = "15th Accuracy"

    Range("FO4").Select
    ActiveCell.FormulaR1C1 = "16th Best method"
    Range("FP4").Select
    ActiveCell.FormulaR1C1 = "16th Hierarchical Method"
    Range("FQ4").Select
    ActiveCell.FormulaR1C1 = "16th Accuracy"

    Range("FR4").Select
    ActiveCell.FormulaR1C1 = "17th Best method"
    Range("FS4").Select
    ActiveCell.FormulaR1C1 = "17th Hierarchical Method"
    Range("FT4").Select
    ActiveCell.FormulaR1C1 = "17th Accuracy"

    Range("FU4").Select
    ActiveCell.FormulaR1C1 = "18th Best method"
    Range("FV4").Select
    ActiveCell.FormulaR1C1 = "18th Hierarchical Method"
    Range("FW4").Select
    ActiveCell.FormulaR1C1 = "18th Accuracy"

    Range("FX4").Select
    ActiveCell.FormulaR1C1 = "19th Best method"
    Range("FY4").Select
    ActiveCell.FormulaR1C1 = "19th Hierarchical Method"
    Range("FZ4").Select
    ActiveCell.FormulaR1C1 = "19th Accuracy"

    Range("GA4").Select
    ActiveCell.FormulaR1C1 = "20th Best method"
    Range("GB4").Select
    ActiveCell.FormulaR1C1 = "20th Hierarchical Method"
    Range("GC4").Select
    ActiveCell.FormulaR1C1 = "20th Accuracy"

    Range("GD4").Select
    ActiveCell.FormulaR1C1 = "21st Best method"
    Range("GE4").Select
    ActiveCell.FormulaR1C1 = "21st Hierarchical Method"
    Range("GF4").Select
    ActiveCell.FormulaR1C1 = "21st Accuracy"

    Range("GG4").Select
    ActiveCell.FormulaR1C1 = "22nd Best method"
    Range("GH4").Select
    ActiveCell.FormulaR1C1 = "22nd Hierarchical Method"
    Range("GI4").Select
    ActiveCell.FormulaR1C1 = "22nd Accuracy"

    Range("GJ4").Select
    ActiveCell.FormulaR1C1 = "23rd Best method"
    Range("GK4").Select
    ActiveCell.FormulaR1C1 = "23rd Hierarchical Method"
    Range("GL4").Select
    ActiveCell.FormulaR1C1 = "23rd Accuracy"

    Range("GM4").Select
    ActiveCell.FormulaR1C1 = "24th Best method"
    Range("GN4").Select
    ActiveCell.FormulaR1C1 = "24th Hierarchical Method"
    Range("GO4").Select
    ActiveCell.FormulaR1C1 = "24th Accuracy"

    Range("GP4").Select
    ActiveCell.FormulaR1C1 = "25th Best method"
    Range("GQ4").Select
    ActiveCell.FormulaR1C1 = "25th Hierarchical Method"
    Range("GR4").Select
    ActiveCell.FormulaR1C1 = "25th Accuracy"

    Range("GS4").Select
    ActiveCell.FormulaR1C1 = "26th Best method"
    Range("GT4").Select
    ActiveCell.FormulaR1C1 = "26th Hierarchical Method"
    Range("GU4").Select
    ActiveCell.FormulaR1C1 = "26th Accuracy"

    Range("GV4").Select
    ActiveCell.FormulaR1C1 = "27th Best method"
    Range("GW4").Select
    ActiveCell.FormulaR1C1 = "27th Hierarchical Method"
    Range("GX4").Select
    ActiveCell.FormulaR1C1 = "27th Accuracy"

    Range("GY4").Select
    ActiveCell.FormulaR1C1 = "28th Best method"
    Range("GZ4").Select
    ActiveCell.FormulaR1C1 = "28th Hierarchical Method"
    Range("HA4").Select
    ActiveCell.FormulaR1C1 = "28th Accuracy"

    Range("HB4").Select
    ActiveCell.FormulaR1C1 = "29th Best method"
    Range("HC4").Select
    ActiveCell.FormulaR1C1 = "29th Hierarchical Method"
    Range("HD4").Select
    ActiveCell.FormulaR1C1 = "29th Accuracy"

    Range("HE4").Select
    ActiveCell.FormulaR1C1 = "30th Best method"
    Range("HF4").Select
    ActiveCell.FormulaR1C1 = "30th Hierarchical Method"
    Range("HG4").Select
    ActiveCell.FormulaR1C1 = "30th Accuracy"

    Range("HH4").Select
    ActiveCell.FormulaR1C1 = "31th Best method"
    Range("HI4").Select
    ActiveCell.FormulaR1C1 = "31th Hierarchical Method"
    Range("HJ4").Select
    ActiveCell.FormulaR1C1 = "31th Accuracy"
    Range("HK4").Select
    ActiveCell.FormulaR1C1 = "32th Best method"
    Range("HL4").Select
    ActiveCell.FormulaR1C1 = "32th Hierarchical Method"
    Range("HM4").Select
    ActiveCell.FormulaR1C1 = "32th Accuracy"
    Range("HN4").Select
    ActiveCell.FormulaR1C1 = "33th Best method"
    Range("HO4").Select
    ActiveCell.FormulaR1C1 = "33th Hierarchical Method"
    Range("HP4").Select
    ActiveCell.FormulaR1C1 = "33th Accuracy"
    Range("HQ4").Select
    ActiveCell.FormulaR1C1 = "34th Best method"
    Range("HR4").Select
    ActiveCell.FormulaR1C1 = "34th Hierarchical Method"
    Range("HS4").Select
    ActiveCell.FormulaR1C1 = "34th Accuracy"
    Range("HT4").Select
    ActiveCell.FormulaR1C1 = "35th Best method"
    Range("HU4").Select
    ActiveCell.FormulaR1C1 = "35th Hierarchical Method"
    Range("HV4").Select
    ActiveCell.FormulaR1C1 = "35th Accuracy"
    Range("HW4").Select
    ActiveCell.FormulaR1C1 = "36th Best method"
    Range("HX4").Select
    ActiveCell.FormulaR1C1 = "36th Hierarchical Method"
    Range("HY4").Select
    ActiveCell.FormulaR1C1 = "36th Accuracy"
    Range("HZ4").Select
    ActiveCell.FormulaR1C1 = "37th Best method"
    Range("IA4").Select
    ActiveCell.FormulaR1C1 = "37th Hierarchical Method"
    Range("IB4").Select
    ActiveCell.FormulaR1C1 = "37th Accuracy"
    Range("IC4").Select
    ActiveCell.FormulaR1C1 = "38th Best method"
    Range("ID4").Select
    ActiveCell.FormulaR1C1 = "38th Hierarchical Method"
    Range("IE4").Select
    ActiveCell.FormulaR1C1 = "38th Accuracy"
    Range("IF4").Select
    ActiveCell.FormulaR1C1 = "39th Best method"
    Range("IG4").Select
    ActiveCell.FormulaR1C1 = "39th Hierarchical Method"
    Range("IH4").Select
    ActiveCell.FormulaR1C1 = "39th Accuracy"
    Range("II4").Select
    ActiveCell.FormulaR1C1 = "40th Best method"
    Range("IJ4").Select
    ActiveCell.FormulaR1C1 = "40th Hierarchical Method"
    Range("IK4").Select
    ActiveCell.FormulaR1C1 = "40th Accuracy"
    Range("IL4").Select
    ActiveCell.FormulaR1C1 = "41th Best method"
    Range("IM4").Select
    ActiveCell.FormulaR1C1 = "41th Hierarchical Method"
    Range("IN4").Select
    ActiveCell.FormulaR1C1 = "41th Accuracy"
    Range("IO4").Select
    ActiveCell.FormulaR1C1 = "42th Best method"
    Range("IP4").Select
    ActiveCell.FormulaR1C1 = "42th Hierarchical Method"
    Range("IQ4").Select
    ActiveCell.FormulaR1C1 = "42th Accuracy"
    Range("IR4").Select
    ActiveCell.FormulaR1C1 = "43th Best method"
    Range("IS4").Select
    ActiveCell.FormulaR1C1 = "43th Hierarchical Method"
    Range("IT4").Select
    ActiveCell.FormulaR1C1 = "43th Accuracy"
    Range("IU4").Select
    ActiveCell.FormulaR1C1 = "44th Best method"
    Range("IV4").Select
    ActiveCell.FormulaR1C1 = "44th Hierarchical Method"
    Range("IW4").Select
    ActiveCell.FormulaR1C1 = "44th Accuracy"
    Range("IX4").Select
    ActiveCell.FormulaR1C1 = "45th Best method"
    Range("IY4").Select
    ActiveCell.FormulaR1C1 = "45th Hierarchical Method"
    Range("IZ4").Select
    ActiveCell.FormulaR1C1 = "45th Accuracy"
    Range("JA4").Select
    ActiveCell.FormulaR1C1 = "46th Best method"
    Range("JB4").Select
    ActiveCell.FormulaR1C1 = "46th Hierarchical Method"
    Range("JC4").Select
    ActiveCell.FormulaR1C1 = "46th Accuracy"
    Range("JD4").Select
    ActiveCell.FormulaR1C1 = "47th Best method"
    Range("JE4").Select
    ActiveCell.FormulaR1C1 = "47th Hierarchical Method"
    Range("JF4").Select
    ActiveCell.FormulaR1C1 = "47th Accuracy"
    Range("JG4").Select
    ActiveCell.FormulaR1C1 = "48th Best method"
    Range("JH4").Select
    ActiveCell.FormulaR1C1 = "48th Hierarchical Method"
    Range("JI4").Select
    ActiveCell.FormulaR1C1 = "48th Accuracy"
    Range("JJ4").Select
    ActiveCell.FormulaR1C1 = "49th Best method"
    Range("JK4").Select
    ActiveCell.FormulaR1C1 = "49th Hierarchical Method"
    Range("JL4").Select
    ActiveCell.FormulaR1C1 = "49th Accuracy"
    Range("JM4").Select
    ActiveCell.FormulaR1C1 = "50th Best method"
    Range("JN4").Select
    ActiveCell.FormulaR1C1 = "50th Hierarchical Method"
    Range("JO4").Select
    ActiveCell.FormulaR1C1 = "50th Accuracy"
    Range("JP4").Select
    ActiveCell.FormulaR1C1 = "51th Best method"
    Range("JQ4").Select
    ActiveCell.FormulaR1C1 = "51th Hierarchical Method"
    Range("JR4").Select
    ActiveCell.FormulaR1C1 = "51th Accuracy"
    Range("JS4").Select
    ActiveCell.FormulaR1C1 = "52th Best method"
    Range("JT4").Select
    ActiveCell.FormulaR1C1 = "52th Hierarchical Method"
    Range("JU4").Select
    ActiveCell.FormulaR1C1 = "52th Accuracy"
    Range("JV4").Select
    ActiveCell.FormulaR1C1 = "53th Best method"
    Range("JW4").Select
    ActiveCell.FormulaR1C1 = "53th Hierarchical Method"
    Range("JX4").Select
    ActiveCell.FormulaR1C1 = "53th Accuracy"
    Range("JY4").Select
    ActiveCell.FormulaR1C1 = "54th Best method"
    Range("JZ4").Select
    ActiveCell.FormulaR1C1 = "54th Hierarchical Method"
    Range("KA4").Select
    ActiveCell.FormulaR1C1 = "54th Accuracy"
    Range("KB4").Select
    ActiveCell.FormulaR1C1 = "55th Best method"
    Range("KC4").Select
    ActiveCell.FormulaR1C1 = "55th Hierarchical Method"
    Range("KD4").Select
    ActiveCell.FormulaR1C1 = "55th Accuracy"
    Range("KE4").Select
    ActiveCell.FormulaR1C1 = "56th Best method"
    Range("KF4").Select
    ActiveCell.FormulaR1C1 = "56th Hierarchical Method"
    Range("KG4").Select
    ActiveCell.FormulaR1C1 = "56th Accuracy"
    Range("KH4").Select
    ActiveCell.FormulaR1C1 = "57th Best method"
    Range("KI4").Select
    ActiveCell.FormulaR1C1 = "57th Hierarchical Method"
    Range("KJ4").Select
    ActiveCell.FormulaR1C1 = "57th Accuracy"
    Range("KK4").Select
    ActiveCell.FormulaR1C1 = "58th Best method"
    Range("KL4").Select
    ActiveCell.FormulaR1C1 = "58th Hierarchical Method"
    Range("KM4").Select
    ActiveCell.FormulaR1C1 = "58th Accuracy"
    Range("KN4").Select
    ActiveCell.FormulaR1C1 = "59th Best method"
    Range("KO4").Select
    ActiveCell.FormulaR1C1 = "59th Hierarchical Method"
    Range("KP4").Select
    ActiveCell.FormulaR1C1 = "59th Accuracy"
    Range("KQ4").Select
    ActiveCell.FormulaR1C1 = "60th Best method"
    Range("KR4").Select
    ActiveCell.FormulaR1C1 = "60th Hierarchical Method"
    Range("KS4").Select
    ActiveCell.FormulaR1C1 = "60th Accuracy"
    Range("KT4").Select
    ActiveCell.FormulaR1C1 = "61th Best method"
    Range("KU4").Select
    ActiveCell.FormulaR1C1 = "61th Hierarchical Method"
    Range("KV4").Select
    ActiveCell.FormulaR1C1 = "61th Accuracy"
    Range("KW4").Select
    ActiveCell.FormulaR1C1 = "62th Best method"
    Range("KX4").Select
    ActiveCell.FormulaR1C1 = "62th Hierarchical Method"
    Range("KY4").Select
    ActiveCell.FormulaR1C1 = "62th Accuracy"
    Range("KZ4").Select
    ActiveCell.FormulaR1C1 = "63th Best method"
    Range("LA4").Select
    ActiveCell.FormulaR1C1 = "63th Hierarchical Method"
    Range("LB4").Select
    ActiveCell.FormulaR1C1 = "63th Accuracy"
    Range("LC4").Select
    ActiveCell.FormulaR1C1 = "64th Best method"
    Range("LD4").Select
    ActiveCell.FormulaR1C1 = "64th Hierarchical Method"
    Range("LE4").Select
    ActiveCell.FormulaR1C1 = "64th Accuracy"
    Range("LF4").Select
    ActiveCell.FormulaR1C1 = "65th Best method"
    Range("LG4").Select
    ActiveCell.FormulaR1C1 = "65th Hierarchical Method"
    Range("LH4").Select
    ActiveCell.FormulaR1C1 = "65th Accuracy"
    Range("LI4").Select
    ActiveCell.FormulaR1C1 = "66th Best method"
    Range("LJ4").Select
    ActiveCell.FormulaR1C1 = "66th Hierarchical Method"
    Range("LK4").Select
    ActiveCell.FormulaR1C1 = "66th Accuracy"
    Range("LL4").Select
    ActiveCell.FormulaR1C1 = "67th Best method"
    Range("LM4").Select
    ActiveCell.FormulaR1C1 = "67th Hierarchical Method"
    Range("LN4").Select
    ActiveCell.FormulaR1C1 = "67th Accuracy"
    Range("LO4").Select
    ActiveCell.FormulaR1C1 = "68th Best method"
    Range("LP4").Select
    ActiveCell.FormulaR1C1 = "68th Hierarchical Method"
    Range("LQ4").Select
    ActiveCell.FormulaR1C1 = "68th Accuracy"
    Range("LR4").Select
    ActiveCell.FormulaR1C1 = "69th Best method"
    Range("LS4").Select
    ActiveCell.FormulaR1C1 = "69th Hierarchical Method"
    Range("LT4").Select
    ActiveCell.FormulaR1C1 = "69th Accuracy"
    Range("LU4").Select
    ActiveCell.FormulaR1C1 = "70th Best method"
    Range("LV4").Select
    ActiveCell.FormulaR1C1 = "70th Hierarchical Method"
    Range("LW4").Select
    ActiveCell.FormulaR1C1 = "70th Accuracy"
    Range("LX4").Select
    ActiveCell.FormulaR1C1 = "71th Best method"
    Range("LY4").Select
    ActiveCell.FormulaR1C1 = "71th Hierarchical Method"
    Range("LZ4").Select
    ActiveCell.FormulaR1C1 = "71th Accuracy"
    Range("MA4").Select
    ActiveCell.FormulaR1C1 = "72th Best method"
    Range("MB4").Select
    ActiveCell.FormulaR1C1 = "72th Hierarchical Method"
    Range("MC4").Select
    ActiveCell.FormulaR1C1 = "72th Accuracy"
    Range("MD4").Select
    ActiveCell.FormulaR1C1 = "73th Best method"
    Range("ME4").Select
    ActiveCell.FormulaR1C1 = "73th Hierarchical Method"
    Range("MF4").Select
    ActiveCell.FormulaR1C1 = "73th Accuracy"
    Range("MG4").Select
    ActiveCell.FormulaR1C1 = "74th Best method"
    Range("MH4").Select
    ActiveCell.FormulaR1C1 = "74th Hierarchical Method"
    Range("MI4").Select
    ActiveCell.FormulaR1C1 = "74th Accuracy"
    Range("MJ4").Select
    ActiveCell.FormulaR1C1 = "75th Best method"
    Range("MK4").Select
    ActiveCell.FormulaR1C1 = "75th Hierarchical Method"
    Range("ML4").Select
    ActiveCell.FormulaR1C1 = "75th Accuracy"
    Range("MM4").Select
    ActiveCell.FormulaR1C1 = "76th Best method"
    Range("MN4").Select
    ActiveCell.FormulaR1C1 = "76th Hierarchical Method"
    Range("MO4").Select
    ActiveCell.FormulaR1C1 = "76th Accuracy"
    Range("MP4").Select
    ActiveCell.FormulaR1C1 = "77th Best method"
    Range("MQ4").Select
    ActiveCell.FormulaR1C1 = "77th Hierarchical Method"
    Range("MR4").Select
    ActiveCell.FormulaR1C1 = "77th Accuracy"
    Range("MS4").Select
    ActiveCell.FormulaR1C1 = "78th Best method"
    Range("MT4").Select
    ActiveCell.FormulaR1C1 = "78th Hierarchical Method"
    Range("MU4").Select
    ActiveCell.FormulaR1C1 = "78th Accuracy"
    Range("MV4").Select
    ActiveCell.FormulaR1C1 = "79th Best method"
    Range("MW4").Select
    ActiveCell.FormulaR1C1 = "79th Hierarchical Method"
    Range("MX4").Select
    ActiveCell.FormulaR1C1 = "79th Accuracy"
    Range("MY4").Select
    ActiveCell.FormulaR1C1 = "80th Best method"
    Range("MZ4").Select
    ActiveCell.FormulaR1C1 = "80th Hierarchical Method"
    Range("NA4").Select
    ActiveCell.FormulaR1C1 = "80th Accuracy"
    Range("NB4").Select
    ActiveCell.FormulaR1C1 = "81th Best method"
    Range("NC4").Select
    ActiveCell.FormulaR1C1 = "81th Hierarchical Method"
    Range("ND4").Select
    ActiveCell.FormulaR1C1 = "81th Accuracy"
    Range("NE4").Select
    ActiveCell.FormulaR1C1 = "82th Best method"
    Range("NF4").Select
    ActiveCell.FormulaR1C1 = "82th Hierarchical Method"
    Range("NG4").Select
    ActiveCell.FormulaR1C1 = "82th Accuracy"
    Range("NH4").Select
    ActiveCell.FormulaR1C1 = "83th Best method"
    Range("NI4").Select
    ActiveCell.FormulaR1C1 = "83th Hierarchical Method"
    Range("NJ4").Select
    ActiveCell.FormulaR1C1 = "83th Accuracy"
    Range("NK4").Select
    ActiveCell.FormulaR1C1 = "84th Best method"
    Range("NL4").Select
    ActiveCell.FormulaR1C1 = "84th Hierarchical Method"
    Range("NM4").Select
    ActiveCell.FormulaR1C1 = "84th Accuracy"
    Range("NN4").Select
    ActiveCell.FormulaR1C1 = "85th Best method"
    Range("NO4").Select
    ActiveCell.FormulaR1C1 = "85th Hierarchical Method"
    Range("NP4").Select
    ActiveCell.FormulaR1C1 = "85th Accuracy"
    Range("NQ4").Select
    ActiveCell.FormulaR1C1 = "86th Best method"
    Range("NR4").Select
    ActiveCell.FormulaR1C1 = "86th Hierarchical Method"
    Range("NS4").Select
    ActiveCell.FormulaR1C1 = "86th Accuracy"
    Range("NT4").Select
    ActiveCell.FormulaR1C1 = "87th Best method"
    Range("NU4").Select
    ActiveCell.FormulaR1C1 = "87th Hierarchical Method"
    Range("NV4").Select
    ActiveCell.FormulaR1C1 = "87th Accuracy"
    Range("NW4").Select
    ActiveCell.FormulaR1C1 = "88th Best method"
    Range("NX4").Select
    ActiveCell.FormulaR1C1 = "88th Hierarchical Method"
    Range("NY4").Select
    ActiveCell.FormulaR1C1 = "88th Accuracy"
    Range("NZ4").Select
    ActiveCell.FormulaR1C1 = "89th Best method"
    Range("OA4").Select
    ActiveCell.FormulaR1C1 = "89th Hierarchical Method"
    Range("OB4").Select
    ActiveCell.FormulaR1C1 = "89th Accuracy"
    Range("OC4").Select
    ActiveCell.FormulaR1C1 = "90th Best method"
    Range("OD4").Select
    ActiveCell.FormulaR1C1 = "90th Hierarchical Method"
    Range("OE4").Select
    ActiveCell.FormulaR1C1 = "90th Accuracy"
    Range("OF4").Select
    ActiveCell.FormulaR1C1 = "91th Best method"
    Range("OG4").Select
    ActiveCell.FormulaR1C1 = "91th Hierarchical Method"
    Range("OH4").Select
    ActiveCell.FormulaR1C1 = "91th Accuracy"
    Range("OI4").Select
    ActiveCell.FormulaR1C1 = "92th Best method"
    Range("OJ4").Select
    ActiveCell.FormulaR1C1 = "92th Hierarchical Method"
    Range("OK4").Select
    ActiveCell.FormulaR1C1 = "92th Accuracy"
    Range("OL4").Select
    ActiveCell.FormulaR1C1 = "93th Best method"
    Range("OM4").Select
    ActiveCell.FormulaR1C1 = "93th Hierarchical Method"
    Range("ON4").Select
    ActiveCell.FormulaR1C1 = "93th Accuracy"
    Range("OO4").Select
    ActiveCell.FormulaR1C1 = "94th Best method"
    Range("OP4").Select
    ActiveCell.FormulaR1C1 = "94th Hierarchical Method"
    Range("OQ4").Select
    ActiveCell.FormulaR1C1 = "94th Accuracy"
    Range("OR4").Select
    ActiveCell.FormulaR1C1 = "95th Best method"
    Range("OS4").Select
    ActiveCell.FormulaR1C1 = "95th Hierarchical Method"
    Range("OT4").Select
    ActiveCell.FormulaR1C1 = "95th Accuracy"
    Range("OU4").Select
    ActiveCell.FormulaR1C1 = "96th Best method"
    Range("OV4").Select
    ActiveCell.FormulaR1C1 = "96th Hierarchical Method"
    Range("OW4").Select
    ActiveCell.FormulaR1C1 = "96th Accuracy"
    Range("OX4").Select
    ActiveCell.FormulaR1C1 = "97th Best method"
    Range("OY4").Select
    ActiveCell.FormulaR1C1 = "97th Hierarchical Method"
    Range("OZ4").Select
    ActiveCell.FormulaR1C1 = "97th Accuracy"
    Range("PA4").Select
    ActiveCell.FormulaR1C1 = "98th Best method"
    Range("PB4").Select
    ActiveCell.FormulaR1C1 = "98th Hierarchical Method"
    Range("PC4").Select
    ActiveCell.FormulaR1C1 = "98th Accuracy"
    Range("PD4").Select
    ActiveCell.FormulaR1C1 = "99th Best method"
    Range("PE4").Select
    ActiveCell.FormulaR1C1 = "99th Hierarchical Method"
    Range("PF4").Select
    ActiveCell.FormulaR1C1 = "99th Accuracy"
    Range("PG4").Select
    ActiveCell.FormulaR1C1 = "100th Best method"
    Range("PH4").Select
    ActiveCell.FormulaR1C1 = "100th Hierarchical Method"
    Range("PI4").Select
    ActiveCell.FormulaR1C1 = "100th Accuracy"

    Range("DP6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(MAX(RC2:RC" & SEMR & "),RC2:RC" & SEMR & ",0)),"""")"
    Range("DQ6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(MAX(RC2:RC" & SEMR & "),RC2:RC" & SEMR & ",0)),"""")"
    Range("DR6").Select
    ActiveCell.Formula2R1C1 = _
        "=MAX(RC2:RC" & SEMR & ")"

    Range("DS6").Select
    ActiveCell.Formula2R1C1 = _
       "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",2),RC2:RC" & SEMR & ",0)),"""")"
    Range("DT6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",2),RC2:RC" & SEMR & ",0)),"""")"
    Range("DU6").Select
    ActiveCell.Formula2R1C1 = _
        "=LARGE(RC2:RC" & SEMR & ",2)"

    Range("DV6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",3),RC2:RC" & SEMR & ",0)),"""")"
    Range("DW6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",3),RC2:RC" & SEMR & ",0)),"""")"
    Range("DX6").Select
    ActiveCell.Formula2R1C1 = _
        "=LARGE(RC2:RC" & SEMR & ",3)"

    Range("DY6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",4),RC2:RC" & SEMR & ",0)),"""")"
    Range("DZ6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",4),RC2:RC" & SEMR & ",0)),"""")"
    Range("EA6").Select
    ActiveCell.Formula2R1C1 = _
        "=LARGE(RC2:RC" & SEMR & ",4)"

    Range("EB6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",5),RC2:RC" & SEMR & ",0)),"""")"
    Range("EC6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",5),RC2:RC" & SEMR & ",0)),"""")"
    Range("ED6").Select
    ActiveCell.Formula2R1C1 = _
        "=LARGE(RC2:RC" & SEMR & ",5)"

    Range("EE6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",6),RC2:RC" & SEMR & ",0)),"""")"
    Range("EF6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",6),RC2:RC" & SEMR & ",0)),"""")"
    Range("EG6").Select
    ActiveCell.Formula2R1C1 = _
        "=LARGE(RC2:RC" & SEMR & ",6)"

    Range("EH6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",7),RC2:RC" & SEMR & ",0)),"""")"
    Range("EI6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",7),RC2:RC" & SEMR & ",0)),"""")"
    Range("EJ6").Select
    ActiveCell.Formula2R1C1 = _
        "=LARGE(RC2:RC" & SEMR & ",7)"

    Range("EK6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",8),RC2:RC" & SEMR & ",0)),"""")"
    Range("EL6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",8),RC2:RC" & SEMR & ",0)),"""")"
    Range("EM6").Select
    ActiveCell.Formula2R1C1 = _
        "=LARGE(RC2:RC" & SEMR & ",8)"

    Range("EN6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",9),RC2:RC" & SEMR & ",0)),"""")"
    Range("EO6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",9),RC2:RC" & SEMR & ",0)),"""")"
    Range("EP6").Select
    ActiveCell.Formula2R1C1 = _
        "=LARGE(RC2:RC" & SEMR & ",9)"

    Range("EQ6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",10),RC2:RC" & SEMR & ",0)),"""")"
    Range("ER6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",10),RC2:RC" & SEMR & ",0)),"""")"
    Range("ES6").Select
    ActiveCell.Formula2R1C1 = _
        "=LARGE(RC2:RC" & SEMR & ",10)"

    Range("ET6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",11),RC2:RC" & SEMR & ",0)),"""")"
    Range("EU6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",11),RC2:RC" & SEMR & ",0)),"""")"
    Range("EV6").Select
    ActiveCell.Formula2R1C1 = _
        "=LARGE(RC2:RC" & SEMR & ",11)"

    Range("EW6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",12),RC2:RC" & SEMR & ",0)),"""")"
    Range("EX6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",12),RC2:RC" & SEMR & ",0)),"""")"
    Range("EY6").Select
    ActiveCell.Formula2R1C1 = _
        "=LARGE(RC2:RC" & SEMR & ",12)"

    Range("EZ6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",13),RC2:RC" & SEMR & ",0)),"""")"
    Range("FA6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",13),RC2:RC" & SEMR & ",0)),"""")"
    Range("FB6").Select
    ActiveCell.Formula2R1C1 = _
        "=LARGE(RC2:RC" & SEMR & ",13)"

    Range("FC6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",14),RC2:RC" & SEMR & ",0)),"""")"
    Range("FD6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",14),RC2:RC" & SEMR & ",0)),"""")"
    Range("FE6").Select
    ActiveCell.Formula2R1C1 = _
        "=LARGE(RC2:RC" & SEMR & ",14)"

    Range("FF6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",15),RC2:RC" & SEMR & ",0)),"""")"
    Range("FG6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",15),RC2:RC" & SEMR & ",0)),"""")"
    Range("FH6").Select
    ActiveCell.Formula2R1C1 = _
        "=LARGE(RC2:RC" & SEMR & ",15)"

    Range("FI6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",16),RC2:RC" & SEMR & ",0)),"""")"
    Range("FJ6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",16),RC2:RC" & SEMR & ",0)),"""")"
    Range("FK6").Select
    ActiveCell.Formula2R1C1 = _
        "=LARGE(RC2:RC" & SEMR & ",16)"


    Range("FL6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",17),RC2:RC" & SEMR & ",0)),"""")"
    Range("FM6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",17),RC2:RC" & SEMR & ",0)),"""")"
    Range("FN6").Select
    ActiveCell.Formula2R1C1 = _
        "=LARGE(RC2:RC" & SEMR & ",17)"


    Range("FO6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",18),RC2:RC" & SEMR & ",0)),"""")"
    Range("FP6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",18),RC2:RC" & SEMR & ",0)),"""")"
    Range("FQ6").Select
    ActiveCell.Formula2R1C1 = _
        "=LARGE(RC2:RC" & SEMR & ",18)"


    Range("FR6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",19),RC2:RC" & SEMR & ",0)),"""")"
    Range("FS6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",19),RC2:RC" & SEMR & ",0)),"""")"
    Range("FT6").Select
    ActiveCell.Formula2R1C1 = _
        "=LARGE(RC2:RC" & SEMR & ",19)"


    Range("FU6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",20),RC2:RC" & SEMR & ",0)),"""")"
    Range("FV6").Select
    ActiveCell.Formula2R1C1 = _
        "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",20),RC2:RC" & SEMR & ",0)),"""")"
    Range("FW6").Select
    ActiveCell.Formula2R1C1 = _
        "=LARGE(RC2:RC" & SEMR & ",20)"

    Range("FX6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",21),RC2:RC" & SEMR & ",0)),"""")"
    Range("FY6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",21),RC2:RC" & SEMR & ",0)),"""")"
    Range("FZ6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",21)"
    Range("GA6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",22),RC2:RC" & SEMR & ",0)),"""")"
    Range("GB6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",22),RC2:RC" & SEMR & ",0)),"""")"
    Range("GC6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",22)"
    Range("GD6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",23),RC2:RC" & SEMR & ",0)),"""")"
    Range("GE6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",23),RC2:RC" & SEMR & ",0)),"""")"
    Range("GF6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",23)"
    Range("GG6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",24),RC2:RC" & SEMR & ",0)),"""")"
    Range("GH6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",24),RC2:RC" & SEMR & ",0)),"""")"
    Range("GI6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",24)"
    Range("GJ6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",25),RC2:RC" & SEMR & ",0)),"""")"
    Range("GK6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",25),RC2:RC" & SEMR & ",0)),"""")"
    Range("GL6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",25)"
    Range("GM6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",26),RC2:RC" & SEMR & ",0)),"""")"
    Range("GN6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",26),RC2:RC" & SEMR & ",0)),"""")"
    Range("GO6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",26)"
    Range("GP6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",27),RC2:RC" & SEMR & ",0)),"""")"
    Range("GQ6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",27),RC2:RC" & SEMR & ",0)),"""")"
    Range("GR6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",27)"
    Range("GS6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",28),RC2:RC" & SEMR & ",0)),"""")"
    Range("GT6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",28),RC2:RC" & SEMR & ",0)),"""")"
    Range("GU6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",28)"
    Range("GV6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",29),RC2:RC" & SEMR & ",0)),"""")"
    Range("GW6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",29),RC2:RC" & SEMR & ",0)),"""")"
    Range("GX6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",29)"
    Range("GY6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",30),RC2:RC" & SEMR & ",0)),"""")"
    Range("GZ6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",30),RC2:RC" & SEMR & ",0)),"""")"
    Range("HA6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",30)"
    Range("HB6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",31),RC2:RC" & SEMR & ",0)),"""")"
    Range("HC6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",31),RC2:RC" & SEMR & ",0)),"""")"
    Range("HD6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",31)"
    Range("HE6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",32),RC2:RC" & SEMR & ",0)),"""")"
    Range("HF6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",32),RC2:RC" & SEMR & ",0)),"""")"
    Range("HG6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",32)"
    Range("HH6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",33),RC2:RC" & SEMR & ",0)),"""")"
    Range("HI6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",33),RC2:RC" & SEMR & ",0)),"""")"
    Range("HJ6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",33)"
    Range("HK6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",34),RC2:RC" & SEMR & ",0)),"""")"
    Range("HL6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",34),RC2:RC" & SEMR & ",0)),"""")"
    Range("HM6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",34)"
    Range("HN6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",35),RC2:RC" & SEMR & ",0)),"""")"
    Range("HO6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",35),RC2:RC" & SEMR & ",0)),"""")"
    Range("HP6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",35)"
    Range("HQ6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",36),RC2:RC" & SEMR & ",0)),"""")"
    Range("HR6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",36),RC2:RC" & SEMR & ",0)),"""")"
    Range("HS6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",36)"
    Range("HT6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",37),RC2:RC" & SEMR & ",0)),"""")"
    Range("HU6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",37),RC2:RC" & SEMR & ",0)),"""")"
    Range("HV6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",37)"
    Range("HW6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",38),RC2:RC" & SEMR & ",0)),"""")"
    Range("HX6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",38),RC2:RC" & SEMR & ",0)),"""")"
    Range("HY6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",38)"
    Range("HZ6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",39),RC2:RC" & SEMR & ",0)),"""")"
    Range("IA6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",39),RC2:RC" & SEMR & ",0)),"""")"
    Range("IB6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",39)"
    Range("IC6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",40),RC2:RC" & SEMR & ",0)),"""")"
    Range("ID6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",40),RC2:RC" & SEMR & ",0)),"""")"
    Range("IE6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",40)"
    Range("IF6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",41),RC2:RC" & SEMR & ",0)),"""")"
    Range("IG6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",41),RC2:RC" & SEMR & ",0)),"""")"
    Range("IH6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",41)"
    Range("II6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",42),RC2:RC" & SEMR & ",0)),"""")"
    Range("IJ6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",42),RC2:RC" & SEMR & ",0)),"""")"
    Range("IK6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",42)"
    Range("IL6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",43),RC2:RC" & SEMR & ",0)),"""")"
    Range("IM6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",43),RC2:RC" & SEMR & ",0)),"""")"
    Range("IN6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",43)"
    Range("IO6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",44),RC2:RC" & SEMR & ",0)),"""")"
    Range("IP6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",44),RC2:RC" & SEMR & ",0)),"""")"
    Range("IQ6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",44)"
    Range("IR6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",45),RC2:RC" & SEMR & ",0)),"""")"
    Range("IS6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",45),RC2:RC" & SEMR & ",0)),"""")"
    Range("IT6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",45)"
    Range("IU6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",46),RC2:RC" & SEMR & ",0)),"""")"
    Range("IV6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",46),RC2:RC" & SEMR & ",0)),"""")"
    Range("IW6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",46)"
    Range("IX6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",47),RC2:RC" & SEMR & ",0)),"""")"
    Range("IY6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",47),RC2:RC" & SEMR & ",0)),"""")"
    Range("IZ6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",47)"
    Range("JA6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",48),RC2:RC" & SEMR & ",0)),"""")"
    Range("JB6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",48),RC2:RC" & SEMR & ",0)),"""")"
    Range("JC6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",48)"
    Range("JD6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",49),RC2:RC" & SEMR & ",0)),"""")"
    Range("JE6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",49),RC2:RC" & SEMR & ",0)),"""")"
    Range("JF6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",49)"
    Range("JG6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",50),RC2:RC" & SEMR & ",0)),"""")"
    Range("JH6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",50),RC2:RC" & SEMR & ",0)),"""")"
    Range("JI6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",50)"
    Range("JJ6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",51),RC2:RC" & SEMR & ",0)),"""")"
    Range("JK6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",51),RC2:RC" & SEMR & ",0)),"""")"
    Range("JL6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",51)"
    Range("JM6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",52),RC2:RC" & SEMR & ",0)),"""")"
    Range("JN6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",52),RC2:RC" & SEMR & ",0)),"""")"
    Range("JO6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",52)"
    Range("JP6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",53),RC2:RC" & SEMR & ",0)),"""")"
    Range("JQ6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",53),RC2:RC" & SEMR & ",0)),"""")"
    Range("JR6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",53)"
    Range("JS6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",54),RC2:RC" & SEMR & ",0)),"""")"
    Range("JT6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",54),RC2:RC" & SEMR & ",0)),"""")"
    Range("JU6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",54)"
    Range("JV6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",55),RC2:RC" & SEMR & ",0)),"""")"
    Range("JW6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",55),RC2:RC" & SEMR & ",0)),"""")"
    Range("JX6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",55)"
    Range("JY6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",56),RC2:RC" & SEMR & ",0)),"""")"
    Range("JZ6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",56),RC2:RC" & SEMR & ",0)),"""")"
    Range("KA6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",56)"
    Range("KB6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",57),RC2:RC" & SEMR & ",0)),"""")"
    Range("KC6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",57),RC2:RC" & SEMR & ",0)),"""")"
    Range("KD6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",57)"
    Range("KE6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",58),RC2:RC" & SEMR & ",0)),"""")"
    Range("KF6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",58),RC2:RC" & SEMR & ",0)),"""")"
    Range("KG6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",58)"
    Range("KH6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",59),RC2:RC" & SEMR & ",0)),"""")"
    Range("KI6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",59),RC2:RC" & SEMR & ",0)),"""")"
    Range("KJ6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",59)"
    Range("KK6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",60),RC2:RC" & SEMR & ",0)),"""")"
    Range("KL6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",60),RC2:RC" & SEMR & ",0)),"""")"
    Range("KM6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",60)"
    Range("KN6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",61),RC2:RC" & SEMR & ",0)),"""")"
    Range("KO6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",61),RC2:RC" & SEMR & ",0)),"""")"
    Range("KP6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",61)"
    Range("KQ6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",62),RC2:RC" & SEMR & ",0)),"""")"
    Range("KR6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",62),RC2:RC" & SEMR & ",0)),"""")"
    Range("KS6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",62)"
    Range("KT6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",63),RC2:RC" & SEMR & ",0)),"""")"
    Range("KU6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",63),RC2:RC" & SEMR & ",0)),"""")"
    Range("KV6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",63)"
    Range("KW6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",64),RC2:RC" & SEMR & ",0)),"""")"
    Range("KX6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",64),RC2:RC" & SEMR & ",0)),"""")"
    Range("KY6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",64)"
    Range("KZ6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",65),RC2:RC" & SEMR & ",0)),"""")"
    Range("LA6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",65),RC2:RC" & SEMR & ",0)),"""")"
    Range("LB6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",65)"
    Range("LC6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",66),RC2:RC" & SEMR & ",0)),"""")"
    Range("LD6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",66),RC2:RC" & SEMR & ",0)),"""")"
    Range("LE6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",66)"
    Range("LF6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",67),RC2:RC" & SEMR & ",0)),"""")"
    Range("LG6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",67),RC2:RC" & SEMR & ",0)),"""")"
    Range("LH6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",67)"
    Range("LI6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",68),RC2:RC" & SEMR & ",0)),"""")"
    Range("LJ6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",68),RC2:RC" & SEMR & ",0)),"""")"
    Range("LK6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",68)"
    Range("LL6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",69),RC2:RC" & SEMR & ",0)),"""")"
    Range("LM6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",69),RC2:RC" & SEMR & ",0)),"""")"
    Range("LN6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",69)"
    Range("LO6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",70),RC2:RC" & SEMR & ",0)),"""")"
    Range("LP6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",70),RC2:RC" & SEMR & ",0)),"""")"
    Range("LQ6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",70)"
    Range("LR6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",71),RC2:RC" & SEMR & ",0)),"""")"
    Range("LS6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",71),RC2:RC" & SEMR & ",0)),"""")"
    Range("LT6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",71)"
    Range("LU6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",72),RC2:RC" & SEMR & ",0)),"""")"
    Range("LV6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",72),RC2:RC" & SEMR & ",0)),"""")"
    Range("LW6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",72)"
    Range("LX6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",73),RC2:RC" & SEMR & ",0)),"""")"
    Range("LY6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",73),RC2:RC" & SEMR & ",0)),"""")"
    Range("LZ6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",73)"
    Range("MA6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",74),RC2:RC" & SEMR & ",0)),"""")"
    Range("MB6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",74),RC2:RC" & SEMR & ",0)),"""")"
    Range("MC6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",74)"
    Range("MD6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",75),RC2:RC" & SEMR & ",0)),"""")"
    Range("ME6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",75),RC2:RC" & SEMR & ",0)),"""")"
    Range("MF6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",75)"
    Range("MG6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",76),RC2:RC" & SEMR & ",0)),"""")"
    Range("MH6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",76),RC2:RC" & SEMR & ",0)),"""")"
    Range("MI6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",76)"
    Range("MJ6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",77),RC2:RC" & SEMR & ",0)),"""")"
    Range("MK6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",77),RC2:RC" & SEMR & ",0)),"""")"
    Range("ML6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",77)"
    Range("MM6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",78),RC2:RC" & SEMR & ",0)),"""")"
    Range("MN6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",78),RC2:RC" & SEMR & ",0)),"""")"
    Range("MO6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",78)"
    Range("MP6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",79),RC2:RC" & SEMR & ",0)),"""")"
    Range("MQ6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",79),RC2:RC" & SEMR & ",0)),"""")"
    Range("MR6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",79)"
    Range("MS6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",80),RC2:RC" & SEMR & ",0)),"""")"
    Range("MT6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",80),RC2:RC" & SEMR & ",0)),"""")"
    Range("MU6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",80)"
    Range("MV6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",81),RC2:RC" & SEMR & ",0)),"""")"
    Range("MW6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",81),RC2:RC" & SEMR & ",0)),"""")"
    Range("MX6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",81)"
    Range("MY6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",82),RC2:RC" & SEMR & ",0)),"""")"
    Range("MZ6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",82),RC2:RC" & SEMR & ",0)),"""")"
    Range("NA6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",82)"
    Range("NB6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",83),RC2:RC" & SEMR & ",0)),"""")"
    Range("NC6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",83),RC2:RC" & SEMR & ",0)),"""")"
    Range("ND6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",83)"
    Range("NE6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",84),RC2:RC" & SEMR & ",0)),"""")"
    Range("NF6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",84),RC2:RC" & SEMR & ",0)),"""")"
    Range("NG6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",84)"
    Range("NH6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",85),RC2:RC" & SEMR & ",0)),"""")"
    Range("NI6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",85),RC2:RC" & SEMR & ",0)),"""")"
    Range("NJ6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",85)"
    Range("NK6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",86),RC2:RC" & SEMR & ",0)),"""")"
    Range("NL6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",86),RC2:RC" & SEMR & ",0)),"""")"
    Range("NM6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",86)"
    Range("NN6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",87),RC2:RC" & SEMR & ",0)),"""")"
    Range("NO6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",87),RC2:RC" & SEMR & ",0)),"""")"
    Range("NP6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",87)"
    Range("NQ6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",88),RC2:RC" & SEMR & ",0)),"""")"
    Range("NR6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",88),RC2:RC" & SEMR & ",0)),"""")"
    Range("NS6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",88)"
    Range("NT6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",89),RC2:RC" & SEMR & ",0)),"""")"
    Range("NU6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",89),RC2:RC" & SEMR & ",0)),"""")"
    Range("NV6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",89)"
    Range("NW6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",90),RC2:RC" & SEMR & ",0)),"""")"
    Range("NX6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",90),RC2:RC" & SEMR & ",0)),"""")"
    Range("NY6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",90)"
    Range("NZ6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",91),RC2:RC" & SEMR & ",0)),"""")"
    Range("OA6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",91),RC2:RC" & SEMR & ",0)),"""")"
    Range("OB6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",91)"
    Range("OC6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",92),RC2:RC" & SEMR & ",0)),"""")"
    Range("OD6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",92),RC2:RC" & SEMR & ",0)),"""")"
    Range("OE6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",92)"
    Range("OF6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",93),RC2:RC" & SEMR & ",0)),"""")"
    Range("OG6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",93),RC2:RC" & SEMR & ",0)),"""")"
    Range("OH6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",93)"
    Range("OI6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",94),RC2:RC" & SEMR & ",0)),"""")"
    Range("OJ6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",94),RC2:RC" & SEMR & ",0)),"""")"
    Range("OK6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",94)"
    Range("OL6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",95),RC2:RC" & SEMR & ",0)),"""")"
    Range("OM6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",95),RC2:RC" & SEMR & ",0)),"""")"
    Range("ON6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",95)"
    Range("OO6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",96),RC2:RC" & SEMR & ",0)),"""")"
    Range("OP6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",96),RC2:RC" & SEMR & ",0)),"""")"
    Range("OQ6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",96)"
    Range("OR6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",97),RC2:RC" & SEMR & ",0)),"""")"
    Range("OS6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",97),RC2:RC" & SEMR & ",0)),"""")"
    Range("OT6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",97)"
    Range("OU6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",98),RC2:RC" & SEMR & ",0)),"""")"
    Range("OV6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",98),RC2:RC" & SEMR & ",0)),"""")"
    Range("OW6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",98)"
    Range("OX6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",99),RC2:RC" & SEMR & ",0)),"""")"
    Range("OY6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",99),RC2:RC" & SEMR & ",0)),"""")"
    Range("OZ6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",99)"
    Range("PA6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",100),RC2:RC" & SEMR & ",0)),"""")"
    Range("PB6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",100),RC2:RC" & SEMR & ",0)),"""")"
    Range("PC6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",100)"
    Range("PD6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",101),RC2:RC" & SEMR & ",0)),"""")"
    Range("PE6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",101),RC2:RC" & SEMR & ",0)),"""")"
    Range("PF6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",101)"
    Range("PG6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R4C2:R4C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",102),RC2:RC" & SEMR & ",0)),"""")"
    Range("PH6").Select
    ActiveCell.Formula2R1C1 = "=IF(ISNUMBER(RC1),INDEX(R5C2:R5C" & SEMR & ",0,MATCH(LARGE(RC2:RC" & SEMR & ",102),RC2:RC" & SEMR & ",0)),"""")"
    Range("PI6").Select
    ActiveCell.Formula2R1C1 = "=LARGE(RC2:RC" & SEMR & ",102)"

    Range("DP6:PI6").Select
    Selection.AutoFill Destination:=Range("DP6:PI880")

    Columns("DP:PI").Select
    With Selection
        .HorizontalAlignment = xlCenter
        .VerticalAlignment = xlBottom
        .WrapText = False
        .Orientation = 0
        .AddIndent = False
        .IndentLevel = 0
        .ShrinkToFit = False
        .ReadingOrder = xlContext
        .MergeCells = False
    End With
    Columns("DP:PI").EntireColumn.AutoFit
    Columns("B:AO").Select
    Range("AO1").Activate
    Selection.ColumnWidth = 11.4
    Range("AO3").Select
    Selection.Copy
    Range("DP5:PI5").Select
    Selection.PasteSpecial Paste:=xlPasteFormats, Operation:=xlNone, _
        SkipBlanks:=False, Transpose:=False
    Application.CutCopyMode = False
    Range("B6").Select
    ActiveWindow.FreezePanes = True
    ActiveWindow.SmallScroll Down:=0
    Columns("A:PI").Select
    sF = "=ISNUMBER($A1)"
    For Each oFC In Selection.FormatConditions
        oFC.Delete
    Next
    Set oFC = Selection.FormatConditions.Add(xlExpression, , sF)
    With oFC
        '设置条件格式的填充色
        .Interior.Color = RGB(221, 235, 247)
    End With

    Sheets("Sheet2").Select
    Sheets("Sheet2").Name = "F1-Score"
End Sub


Public Function getSheetName() As String

     getSheetName = ActiveWindow.ActiveSheet.Name
End Function


