from marshmallow import Schema, fields


class InputSchema(Schema):
    riskScore = fields.Int()
    amountToInvest = fields.Float()


class InputSchemaTwitter(Schema):
    stock_symbol = fields.Str()
    start_date = fields.Date()
    end_date = fields.Date()
