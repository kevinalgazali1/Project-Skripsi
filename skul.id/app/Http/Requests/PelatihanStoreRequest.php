<?php

namespace App\Http\Requests;

use Illuminate\Foundation\Http\FormRequest;

class PelatihanStoreRequest extends FormRequest
{
    /**
     * Determine if the user is authorized to make this request.
     */
    public function authorize(): bool
    {
        return true;
    }

    /**
     * Get the validation rules that apply to the request.
     *
     * @return array<string, \Illuminate\Contracts\Validation\ValidationRule|array<mixed>|string>
     */
    public function rules(): array
    {
        return [
            'nama_pelatihan' => 'required',
            'tempat_pelatihan' => 'required',
            'tanggal_mulai' => 'required',
            'tanggal_selesai' => 'required',
            'deskripsi' => 'required',
            'biaya' => 'nullable',
            'kota' => 'required',
            'status' => 'required|in:Gratis,Berbayar',
            'foto_pelatihan' => 'required|image|mimes:jpeg,png,jpg|max:10048',
            'nomor_rekening' => 'nullable',
        ];
    }
}
